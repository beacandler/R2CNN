
# R<sup>2</sup>CNN: Rotational Region CNN for Orientation Robust Scene Text Detection

## Abstract
This is a caffe re-implementation of [R<sup>2</sup>CNN: Rotational Region CNN for Orientation Robust Scene Text Detection](https://arxiv.org/abs/1706.09579).

This project is modified from [py-R-FCN](https://github.com/YuwenXiong/py-R-FCN), and [inclined nms](./src/lib/lanms) and [generate rotated box](./src/lib/fast_rcnn/icdar.py) component is imported from [EAST project](https://github.com/argman/EAST).
Thanks for the author's([@zxytim](https://github.com/zxytim) [@argman](https://github.com/argman)) help. Please cite [this paper](https://arxiv.org/abs/1704.03155v2) if you find this useful.

## Contents
1. [Abstract](##Abstract)
2. [Structor](#Structor)
3. [Installation](#Installation)
4. [Demo](#Demo)
5. [Test](#Test)
6. [Train](#Train)
7. [Experiments](#Experiments)
8. [Furthermore](#Furthermore)


## Structor
### Code structor
```
.
├── docker-compose.yml
├── docker // docker deps file
├── Dockerfile // docker build file
├── model // model directory
│   ├── caffemodel // trained caffe model
│   ├── icdar15_gt // ICDAR2015 groundtruth
│   ├── prototxt // caffe prototxt file
│   └── imagenet_models // pretrained on imagenet
├── nvidia-docker-compose.yml
├── logs
│   ├── submit // original submit file
│   ├── submit_zip // zip submit file
│   ├── snapshots
│   └── train
│       ├── VGG16.txt.*
│       └── snapshots
├── README.md
├── requirements.txt // python package
├── src
│   ├── cfgs // train config yml
│   ├── data // cache file
│   ├── lib
│   ├── _init_path.py
│   ├── demo.py
│   ├── eval_icdar15.py // eval 2015 icdar dataset F-meaure
│   ├── test_net.py
│   └── train_net.py
├── demo.sh
├── train.sh
├── images // test images
│   ├── img_1.jpg
│   ├── img_2.jpg
│   ├── img_3.jpg
│   ├── img_4.jpg
│   └── img_5.jpg
└── test.sh // test script
```
### Data structor
It should have this basic structure
```
ICDARdevkit_Root
.
├── ICDAR2013
├── merge_train.txt  // images list contains ICDAR2013+ICDAR2015 train dataset, then raw data augmentation the same as the paper
├── ICDAR2015
│   ├── augmentation // contains all augmented images
│   └── ImageSets/Main/test.txt // ICDAR2015 test images list
```
## Installation
### Install caffe
It is highly recommended to use docker to build environment. More about how to configure docker, see [Running with Docker](https://github.com/beacandler/tf-slim-demo#Running)
If you are familiar with docker, please run
```
    1. nvidia-docker-compose run --rm --service-ports rrcnn bash
    2. bash ./demo.sh
```
If you don't familiar with docker, please follow [py-R-FCN](https://github.com/YuwenXiong/py-R-FCN) to install caffe.
### Build
```
    cd src/lib && make
    
```
### Download Model
1. please download [VGG16 pre-trained model](https://pan.baidu.com/s/1Pok-AYU0Jl-DNKrSqF3vNg#list/path=%2FRRCNN%2Fmodel%2Fimagenet_models) on Imagenet, place it to model/imagenet_models/VGG16.v2.caffemodel.
2. please download [VGG16 trained model](https://pan.baidu.com/s/1Pok-AYU0Jl-DNKrSqF3vNg#list/path=%2FRRCNN%2Fmodel%2Fcaffemodel) by this project, place it model/caffemodel/TextBoxes-v2_iter_12w.caffemodel.
 
## Demo
It is recommended to use UNIX socket to support GUI for docker, plesase open another terminal and type:
```bash
    xhost + # may be you need it when open a new terminal
    # docker-compose.yml: mount host  volume : /tmp/.X11-unix to docker volume: /tmp/.X11-unix  
    # pass DISPLAY variable to docker container so host X server can display image in docker
    docker exec -it -e DISPLAY=$DISPLAY ${CURRENT_CONTAINER_ID} bash
    bash ./demo.sh
```

## Test
### Single Test
```bash
    bash ./test.sh
```
### Multi-scale Test


```bash
    # please uncomment two lines in src/cfgs/faster_rcnn_end2end.yml
    SCALES: [720, 1200]
    MULTI_SCALES_NOC: True
    # modify src/lib/datasets/icdar.py to find ICDAR2015 test data, please refer to commit @bbac1cf
    # then run
    bash ./test.sh
```
## Train
### Train data
> * Mine: ICDAR2013+ICDAR2015 train dataset, and raw data augmentation, at last got 15977 images.
> * Paper: ICDAR2015 + 2000 focused scene text images they collected.

### Train commands
1. Go to ./src/lib/datasets/icdar.py, modify images path to let train.py find merge_train.txt images list.
2. Remove cache in src/data/*.pkl or you can load cached [roidb data](https://pan.baidu.com/disk/home?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#list/vmode=list&path=%2FRRCNN%2Fcache_roidb_data) of this project, and place it to src/data/
3. 
```bash
    # Train for RRCNN4-TextBoxes-v2-OHEM
    bash ./train.sh
```
note: If you use USE_FLIPPED=True&USE_FLIPPED_QUAD=True, you will get almost 31200 roidb.
## Experiments

### Mine VS Paper

|Approaches|Anchor Scales|Pooled sizes|Inclined NMS|Test scales(short side)|F-measure(Mine VS paper)|
|-------------------|:---------------------:|:-----:|:------------------:|:------------------:|:------------------:|
|R<sup>2</sup>CNN-2 | (4, 8, 16)   | (7, 7) |Y|(720)|71.12% VS 68.49%|           
|R<sup>2</sup>CNN-3 | (4, 8, 16)   | (7, 7) |Y|(720)|73.10% VS 74.29%|           
|R<sup>2</sup>CNN-4 | (4, 8, 16, 32)| (7, 7) |Y|(720)|74.14% VS 74.36%|           
|R<sup>2</sup>CNN-4 | (4, 8, 16, 32)| (7, 7) |Y|(720, 1200)|79.05% VS 81.80%|           
|R<sup>2</sup>CNN-5 | (4, 8, 16, 32)| (7, 7) (11, 3) (3, 11) |Y|(720)|74.34% VS 75.34%|            
|R<sup>2</sup>CNN-5 | (4, 8, 16, 32)| (7, 7) (11, 3) (3, 11) |Y|(720, 1200)|78.70% VS 82.54%|              

### Appendixes


|Approaches      | Anchor Scales | aspect ration| Pooled sizes | Inclined NMS| Test scales(short side)| F-measure|
|-------------------|:-------------------:|:---------------------:|:-----:|:------------------:|:------------------:|:------------------:|
|R<sup>2</sup>CNN-4 | (4, 8, 16, 32)|(0.5, 1, 2)| (7, 7) |Y|(720)|74.36%|           
|R<sup>2</sup>CNN-4 | (4, 8, 16, 32)|(0.5, 1, 2)| (7, 7) |Y|(720, 1200)|VS 81.80%|           
|R<sup>2</sup>CNN-4-TextBoxes-OHEM | (4, 8, 16, 32)|(0.5, 1, 2, 3, 5, 7, 10)| (7, 7) |Y|(720)|76.53%|          

## Furthermore

You can try Resnet-50, Resnet-101 and so on.
