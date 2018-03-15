#!/bin/bash
# Usage:
# ./test.sh \

export PYTHONUNBUFFERED="True"
GPU_ID=1

caffemodel='model/caffemodel/TextBoxes-v2_iter_12w.caffemodel'
caffemodel='logs/train/snapshots/_iter_100000.caffemodel'

CUDA_VISIBLE_DEVICES=${GPU_ID}
time ./src/test_net.py --gpu ${GPU_ID} \
  --def model/prototxt/test/TextBoxes-v3.prototxt \
  --net ${caffemodel} \
  --imdb icdar_2015_test \
  --cfg src/cfgs/faster_rcnn_end2end.yml \
  --submit_dir /app/logs/submit \
  --submit_prefix res_

for confidence in {4..9};
do
    echo "confidence: 0."$confidence
    python ./src/eval_icdar15.py -g=./model/icdar15_gt/gt.zip -s=./logs/submit_zip/submit-0.$confidence
done
