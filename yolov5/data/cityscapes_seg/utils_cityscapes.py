import cv2
import numpy as np
import os
import shutil
from collections import defaultdict
from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import tqdm


def img_label2masks(path_to_label: str, ids_instanceonly: list):
    label = Image.open(path_to_label)
    label, instances = np.array(label), np.unique(label)

    classes_unscaled = np.array([ins for ins in instances if ins // 1000 in ids_instanceonly])
    masks = (label == classes_unscaled[:, None, None]).astype(np.uint8)

    return masks, classes_unscaled // 1000, label.shape


def mask2segments(mask: np.array):
    segments = []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour = contour.reshape(-1, 2)
        if contour.shape[0] > 2:
            # the contour should contain at least 3 points
            segments.append(contour)
    if len(segments) == 0:
        segments.append(np.zeros((0, 2)))

    return segments


def segments2polygon(segments: list):
    """Connect segments into one polygon"""
    largest_segm_ind = np.argmax([seg.shape[0] for seg in segments])
    largest_segment = segments[largest_segm_ind]

    for i, curr_segment in enumerate(segments):
        if i == largest_segm_ind:
            continue
        # find nearest points (pt_1, pt_2) between the current segment and the largest one
        dist = cdist(curr_segment, largest_segment)
        idx_curr, idx_largest = np.unravel_index(dist.argmin(), dist.shape)
        # then connect them as follows
        # <- pt_1 -> <- segment_1(..., pt_1) -> <- segment_2(pt_2, ...) -> <- pt_2 ->
        segs_to_connect = [largest_segment[idx_largest][None],
                           np.roll(largest_segment[::-1], idx_largest, axis=0),
                           np.roll(curr_segment, -idx_curr, axis=0),
                           curr_segment[idx_curr][None]
                           ]

        largest_segment = np.concatenate(segs_to_connect)

    return largest_segment


def create_train_val_split(data_dir):
    images, targets = defaultdict(list), defaultdict(list)

    for split in ['train', 'val']:
        images_dir = os.path.join(data_dir, f'leftImg8bit/{split}')
        targets_dir = os.path.join(data_dir, f'gtFine/{split}')

        for city in os.listdir(images_dir):
            img_dir = os.path.join(images_dir, city)
            target_dir = os.path.join(targets_dir, city)

            for file_name in os.listdir(img_dir):
                target_name = f'{file_name.split("_leftImg8bit")[0]}{"_gtFine_instanceIds.png"}'
                targets[split].append(os.path.join(target_dir, target_name))
                images[split].append(os.path.join(img_dir, file_name))

    return images, targets


def prepare_ids(images, targets, num_f, num_w):

    zipped = list(zip(images['train'], targets['train']))
    rng = np.random.default_rng(42)
    rng.shuffle(zipped)
    (images['train'], targets['train']) = zip(*zipped)

    # split on train and train_weak set
    images['train_weak'], targets['train_weak'] = (images['train'][num_f:num_f + num_w],
                                                   targets['train'][num_f:num_f + num_w])
    images['train'], targets['train'] = images['train'][:num_f], targets['train'][:num_f]

    return images, targets


def convert_cityscapes(data_dir, images, targets, ids_instanceonly, new_labels):
    for split in ['train', 'train_weak', 'val']:
        imgs_dir_new = f'{data_dir}/images/{split}'
        targets_dir_new = f'{data_dir}/labels/{split}'
        os.makedirs(imgs_dir_new, exist_ok=True)
        os.makedirs(targets_dir_new, exist_ok=True)

        for img_path, target_path in tqdm(zip(images[split], targets[split]), desc=f'{split}',
                                          total=len(images[split])):
            seg_masks, classes, (h, w) = img_label2masks(target_path, ids_instanceonly)
            classes = [new_labels[old_label] for old_label in classes]
            file_name = img_path.split("/")[-1]
            shutil.copy(img_path, f'{imgs_dir_new}/{file_name}')

            with open(f'{targets_dir_new}/{file_name.split(".png")[0]}.txt', 'w') as f:
                for mask, cls in zip(seg_masks, classes):
                    segments = mask2segments(mask)
                    polygon = segments2polygon(segments)
                    if polygon.shape[0] == 0:
                        continue

                    if split == 'train_weak':
                        xmin, ymin, wi, he = cv2.boundingRect(polygon)
                        polygon = np.array([xmin, ymin, xmin + wi - 1, ymin, xmin + wi - 1, ymin + he - 1, xmin,
                                            ymin + he - 1]).reshape(-1, 2)

                    polygon = (polygon / np.array([w, h])).reshape(-1)  # normalize by img size
                    polygon = np.round(polygon, 6).astype(str)
                    cls_and_poly = f'{cls} ' + ' '.join(polygon) + '\n'
                    f.write(cls_and_poly)