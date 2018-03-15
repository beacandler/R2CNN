#!/usr/bin/env bash
# Usage:
# ./train.sh \

mkdir logs/train

export PYTHONUNBUFFERED="True"

GPU_ID=1

LOG="logs/train/VGG16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


CUDA_VISIBLE_DEVICES=${GPU_ID}
time ./src/train_net.py --gpu ${GPU_ID} \
  --solver model/prototxt/solver.prototxt \
  --weights model/imagenet_models/VGG16.v2.caffemodel \
  --imdb icdar_2015_train \
  --iters 200000 \
  --cfg src/cfgs/faster_rcnn_end2end_ohem.yml \
