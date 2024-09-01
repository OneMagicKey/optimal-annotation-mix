import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from scipy.spatial.distance import cdist
from PIL import Image
from shutil import copy
from tqdm import tqdm


def convert_label(path, lb_path, image_id, names):
    def convert_box(size, box):
        w, h = size
        xmin, xmax, ymin, ymax = box
        xmin, xmax, ymin, ymax = xmin / w, xmax / w, ymin / h, ymax / h
        return xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax

    with open(path / 'Annotations' / f'{image_id}.xml', 'r') as in_file:
        tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(lb_path, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls in names and int(obj.find('difficult').text) != 1:
                xmlbox = obj.find('bndbox')
                bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in
                                          ('xmin', 'xmax', 'ymin', 'ymax')])
                cls_id = names.index(cls)  # class id
                out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

    if os.path.getsize(lb_path) == 0:  # ignore images with no objects
        os.remove(lb_path)


def img_label2masks(path_to_label: str, classes: list):
    label = Image.open(path_to_label)
    label = np.array(label)
    obj_ids = np.arange(1, len(classes) + 1)
    masks = (label == obj_ids[:, None, None]).astype(np.uint8)

    return masks[np.array(classes) != -1], label.shape


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


def get_class_label(path: str, image_id: str, names: list):
    in_file = open(path / f'Annotations/{image_id}.xml')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    classes = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in names:
            classes.append(-1)  # ignore class
            continue

        cls_id = names.index(cls)
        classes.append(cls_id)

    return classes


def prepare_indices(path, num_f, num_w=None):
    trainval_bbox_ids = open(path / 'ImageSets/Main/trainval.txt', 'r').read().strip().split()  # imgs with bbox annotation
    train_seg_ids = open(path / 'ImageSets/Segmentation/train.txt', 'r').read().strip().split()
    val_seg_ids = open(path / 'ImageSets/Segmentation/val.txt', 'r').read().strip().split()
    train_weak_ids = sorted(set(trainval_bbox_ids).difference(set(val_seg_ids)).difference(set(train_seg_ids)))

    rng = np.random.default_rng(42)
    rng.shuffle(train_seg_ids)
    rng.shuffle(train_weak_ids)

    train_seg_ids, unused_ids, train_weak_ids = train_seg_ids[:num_f], train_seg_ids[num_f:], train_weak_ids[:num_w]
    if num_w is None:
        train_weak_ids += unused_ids  # add unused images from the F set to the W set

    ids = {'train': train_seg_ids,
           'train_weak': train_weak_ids,
           'val': val_seg_ids
           }

    return ids


def convert_voc(data_dir, path, ids, names):
    for split in ['train', 'train_weak', 'val']:
        imgs_path = data_dir / 'images' / split
        lbs_path = data_dir / 'labels' / split
        imgs_path.mkdir(exist_ok=True, parents=True)
        lbs_path.mkdir(exist_ok=True, parents=True)

        for img_id in tqdm(ids[split], desc=split):
            old_img_path = path / 'JPEGImages' / f'{img_id}.jpg'
            copy(old_img_path, imgs_path / f'{img_id}.jpg')
            if split == 'train_weak':
                lb_path = lbs_path / f'{img_id}.txt'
                convert_label(path, lb_path, img_id, names)
                continue

            label_path = path / 'SegmentationObject' / f'{img_id}.png'
            class_labels = get_class_label(path, img_id, names)
            seg_masks, (h, w) = img_label2masks(label_path, class_labels)

            class_labels = np.array(class_labels)
            class_labels = class_labels[class_labels != -1]
            assert seg_masks.shape[0] == class_labels.shape[0], 'number of class labels should be equal to number of masks'

            with open(lbs_path / f'{img_id}.txt', 'w') as f:
                for mask, cls in zip(seg_masks, class_labels):
                    segments = mask2segments(mask)
                    polygon = segments2polygon(segments)
                    if polygon.shape[0] == 0:
                        continue

                    polygon = (polygon / np.array([w, h])).reshape(-1)  # normalize by img size
                    polygon = np.round(polygon, 6).astype(str)
                    cls_and_poly = f'{cls} ' + ' '.join(polygon) + '\n'
                    f.write(cls_and_poly)
