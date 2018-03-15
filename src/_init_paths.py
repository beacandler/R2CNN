# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, 'lib')
sys.path.insert(0, lib_path)
