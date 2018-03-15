#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import fastnms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse


CLASSES = ('__background__', 'text')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(image_name, im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        if cfg.TEST.BBOX_REG or not cfg.TEST.USE_INCLINED_NMS:
            bbox = dets[i, :4]
            score = dets[i, -1]
            if score > thresh:
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='red', linewidth=3.5)
                )
        elif cfg.TEST.INCLINED_RECT_REG:
            bbox = dets[i, :8]
            score = dets[i, -1]
            if score > thresh:
                ax.add_patch(
                    plt.Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                 (bbox[4], bbox[5]), (bbox[6], bbox[7])], fill=False,
                                  edgecolor='red', linewidth=3.5)
                    )
        ax.text(bbox[0]+100, bbox[1] - 2 + 100,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes= im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.1
    if cfg.TEST.BBOX_REG:
        step = 4
    elif cfg.TEST.INCLINED_RECT_REG:
        step = 8
    else:
        print 'please config BBOX_REG or '
        return
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, step*cls_ind:step*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        if cfg.TEST.BBOX_REG:
            assert not cfg.TEST.INCLINED_RECT_REG
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
        elif cfg.TEST.INCLINED_RECT_REG:
            assert not cfg.TEST.BBOX_REG
            if cfg.TEST.USE_INCLINED_NMS:
                # dets = standard_nms(dets.astype(np.float64), NMS_THRESH)
                dets = fastnms.standard_nms_n9(dets.astype('float32'), NMS_THRESH)
            else:
                dets_AABB = np.zeros((dets.shape[0], 5), dets.dtype)
                # score
                dets_AABB[:, 4] = dets[:, 8]
                dets_AABB[:, 0] = np.min(dets[:, :8][:, 0::2], axis=1)
                dets_AABB[:, 1] = np.min(dets[:, :8][:, 1::2], axis=1)
                dets_AABB[:, 2] = np.max(dets[:, :8][:, 0::2], axis=1)
                dets_AABB[:, 3] = np.max(dets[:, :8][:, 1::2], axis=1)
                keep = nms(dets_AABB, NMS_THRESH)
                dets = dets_AABB[keep, :]
        vis_detections(image_name, im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.INCLINED_RECT_REG = True
    cfg.TEST.USE_INCLINED_NMS = True
    cfg.TEST.MAX_SIZE = 5000
    cfg.TEST.MULTI_SCALES_NOC = True
    cfg.TEST.SCALES = (720,  1200)
    # cfg.TEST.SCALES = (1000, 1100)
    # cfg.TEST.BBOX_REG = True

    args = parse_args()

    prototxt = '/app/model/prototxt/test/TextBoxes-v3.prototxt'
    caffemodel = '/app/model/caffemodel/TextBoxes-v2_iter_12w.caffemodel'
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _= im_detect(net, im)

    im_names = os.listdir('/app/images')
    im_names = [os.path.join('/app/images', image) for image in im_names]
    count = 0
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
