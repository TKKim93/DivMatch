# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
import os
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_integrated import pascal_voc_integrated
from datasets.voc_clipart import voc_clipart
from datasets.voc_watercolor import voc_watercolor
from datasets.voc_comic import voc_comic

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, devkit_path='../faster-rcnn.pytorch/data/VOCdevkit2007'))

# clipart
for split in ['trainval']:
    for shift in ['CP', 'R', 'CPR']:
        name = 'clipart_{}_{}'.format(shift, split)
        __sets[name] = (lambda split=split, year=year: voc_clipart(split, year, devkit_path=os.path.join('datasets/', 'clipart_{}'.format(shift))))
    name = 'clipart_{}'.format(shift, split)

__sets[name] = (lambda split=split, year=year: voc_clipart(split, year, devkit_path=os.path.join('datasets/', 'clipart_{}'.format(shift))))

# watercolor
for split in ['trainval']:
    for shift in ['CP', 'R', 'CPR']:
        name = 'watercolor_{}_{}'.format(shift, split)
        __sets[name] = (lambda split=split, year=year: voc_watercolor(split, year, devkit_path=os.path.join('datasets/', 'watercolor_{}'.format(shift))))
    name = 'watercolor_{}'.format(shift, split)

# comic
for split in ['trainval']:
    for shift in ['CP', 'R', 'CPR']:
        name = 'comics_{}_{}'.format(shift, split)
        __sets[name] = (lambda split=split, year=year: voc_comic(split, year, devkit_path=os.path.join('datasets/', 'comic_{}'.format(shift))))
    name = 'comic_{}'.format(shift, split)

# Set up voc_integrated
for split in ['trainval']:
  name = 'voc_integrated_{}'.format(split)
  __sets[name] = (lambda split=split: pascal_voc_integrated(split, devkit_path='datasets/Pascal/VOCdevkit/VOC_Integrated'))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  # print(__sets[name]())
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
