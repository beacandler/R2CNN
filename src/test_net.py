#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=400, type=int)
    parser.add_argument('--rpn_file', dest='rpn_file',
                        default=None, type=str)
    parser.add_argument('--submit_dir', help='Directory where to store submit results files',
                        default='/app/logs/submit', type=str)
    parser.add_argument('--submit_prefix', help='submit format prefix ',
                        default='', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def result_to_submit(submit_dir):
    """
    1. use py-R-FCN to produce bbs , firstly conf=-np.conf, second nms
    2. then use this func to save submit-{conf}.zip
    3. assume that erery line is
    767,412,822,412,822,440,767,440,score"""

    # attention: must modify write_boxes_file func
    submit_zip_dir = os.path.join(submit_dir.rsplit('/', 1)[0], 'submit_zip')
    if not os.path.exists(submit_zip_dir):
        os.makedirs(submit_zip_dir)
    confs = np.arange(0.4, 1, 0.1)
    major_list = os.listdir(submit_dir)
    import zipfile
    import shutil
    for conf in confs:
        zip_file_name = os.path.join(submit_zip_dir, 'submit-{}'.format(conf))
        zip_file = zipfile.ZipFile(zip_file_name, 'w' )
        save_dir = os.path.join(submit_zip_dir, 'submit_{}'.format(conf))
        for major in major_list:
            gt_path = os.path.join(submit_dir, major)
            filtered_lines = [line.strip() for line in open(gt_path).readlines() if
                              float(line.strip().split(',')[-1]) > conf]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            f_txt = open(os.path.join(save_dir, major), 'w')
            for line in filtered_lines:
                f_txt.write(line.rsplit(',', 1)[0]+'\n')
            f_txt.close()
            zip_file.write(os.path.join(save_dir, major),  os.path.basename(os.path.join(save_dir, major)))
        zip_file.close()
        shutil.rmtree(save_dir)


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)

    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
        if cfg.TEST.PROPOSAL_METHOD == 'rpn':
            imdb.config['rpn_file'] = args.rpn_file

    test_net(net, imdb, max_per_image=args.max_per_image, vis=False,
             submit_dir=args.submit_dir,
             submit_prefix=args.submit_prefix)
    result_to_submit(args.submit_dir)