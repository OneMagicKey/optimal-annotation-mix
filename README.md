# Balancing Bounding Box and Mask Annotations for Semi-Supervised Instance Segmentation

## Run

Here we present the scripts to run experiments with **YOLOv5** model. For the **DETR** model, 
see jupyter notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OneMagicKey/optimal-annotation-mix/blob/master/run.ipynb)


### Data splits
To emulate different setups, we created *.yaml* files with different data splits and placed them 
in _yolov5/data_:
```
├── yolov5
    └── data
        ├── cityscapes_seg  # Cityscapes yamls
        └── voc_seg         # PascalVOC yamls
```

YAML files have the following naming conventions *{dataset}-seg-fw-{**F** size}-{**W** size}.yaml*, 
i.e. *voc-seg-fw-400-3000.yaml* is a _PascalVOC_ split with 400 fully annotated images and 3000 
weakly annotated images. YAMLs without the **W** parameter use all non-**F** data as weak images.

### f-size parameter
The *f-size* parameter controls the number of fully annotated images in the batch. For example, 
if *batch-size = 16* and *f-size = 4*, the resulting batch would contain *4* fully annotated images 
and *12* weakly annotated images. To train on full images only, you can use *f-size=-1* (default setup).

### YOLO training
Clone the repo
```bash
git clone https://github.com/OneMagicKey/optimal-annotation-mix.git
cd optimal-annotation-mix/yolov5
```
and install all the dependencies:
```bash
pip install -r requirements.txt 
```
or create conda env instead:
```bash
conda create -n optimal-annotation-mix
conda activate optimal-annotation-mix
pip install -r requirements.txt
```
#### YOLO PascalVOC

PascalVOC2012 will be downloaded automatically

The following command runs the training of the YOLO model on the PascalVOC dataset with 800 fully annotated and 3000 weakly 
annotated images for 75 epochs/30k iterations.

```bash
python segment/train.py --img 512 --batch-size 16 --f-size 2 --epochs 75 --data voc-seg-fw-800-3000.yaml --hyp hyp.scratch-low.yaml --weights yolov5x-cls.pt --cfg yolov5x-seg.yaml --cache --no-overlap --patience 0
```

#### YOLO Cityscapes
Download the Cityscapes dataset and place it in the datasets directory with 
the following structure

```
├── yolov5
└── datasets
    └── Cityscapes
        └── data
            ├── gtFine
            └── leftImg8bit
```

or download it from [Google Drive](https://drive.google.com/file/d/1Jb2rGbcOOykIDpoqlSZQXg-DsmC9DGdG/view?usp=drive_link)
and extract it with the command:

```bash
# yolov5 is the current dir
path_to_cityscapes = 'path/to/Cityscapes.zip'
cd .. &&  mkdir -p 'datasets/Cityscapes' && unzip -nq $path_to_cityscapes -d 'datasets/' && cd yolov5
```

Then run the training:
```bash
python segment/train.py --img 1024 --batch-size 8 --f-size 4 --epochs 136 --data cityscapes_seg/Cityscapes-seg-fw-1475-1000.yaml --hyp hyp.scratch-low.yaml --weights yolov5x-cls.pt --cfg yolov5x-seg.yaml --cache --no-overlap --patience 0
```

## Acknowledgement
This code is based on [YOLOv5](https://github.com/ultralytics/yolov5) and [DETR](https://github.com/facebookresearch/detr) repos
