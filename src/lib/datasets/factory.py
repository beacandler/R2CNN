# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.icdar import icdar

# Set up icdar_<year>_<split>
for year in ['2015']:
    for split in ['train', 'test']:
    # for split in ['train']:
        name = 'icdar_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: icdar(split, year))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
