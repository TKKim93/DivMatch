# --------------------------------------------------------
# Pytorch Diversify and Match
# Licensed under The MIT License [see LICENSE for details]
# Written by Taeykyung Kim based on code from Jiasen Lu, Jianwei Yang, and Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_da_data_layer.roidb import combined_roidb
from roi_da_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient
from model.faster_rcnn.DDMRL_vgg16 import vgg16

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default="pascal_voc", type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res101', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=16, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=10, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=34, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=9, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=10022, type=int)
# log and diaplay
  parser.add_argument('--use_tfboard', dest='use_tfboard',
                      help='whether use tensorflow tensorboard',
                      default=False, type=bool)

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.use_tfboard:
    from model.utils.logger import Logger
    # Set the logger
    logger = Logger('./logs')

  if args.dataset == "clipart":
      args.imdb_name = "voc_integrated_trainval"
      args.imdbval_name = "clipart_train"
      args.imdbguide_name = "ClipCP_trainval"
      args.imdbguide2_name = "ClipR_trainval"
      args.imdbguide3_name = "ClipCPR_trainval"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20'
  elif args.dataset == "watercolor":
      args.imdb_name = "voc_integrated_trainval"
      args.imdbval_name = "watercolor_train"
      args.imdbguide_name = "WatCP_trainval"
      args.imdbguide2_name = "WatR_trainval"
      args.imdbguide3_name = "WatCPR_trainval"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "comic":
      args.imdb_name = "voc_integrated_trainval"
      args.imdbval_name = "clipart_train"
      args.imdbguide_name = "ComCP_trainval"
      args.imdbguide2_name = "ComR_trainval"
      args.imdbguide3_name = "ComCPR_trainval"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "cityscapes":
      args.imdb_name = "cityscapes_train"
      args.imdbval_name = "foggy_cityscapes_val"
      args.imdbguide_name = "CityCP_train"
      args.imdbguide2_name = "CityR_train"
      args.imdbguide3_name = "CityCPR_train"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda

  # SOURCE
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # TARGET
  imdb2, roidb2, ratio_list2, ratio_index2 = combined_roidb(args.imdbval_name)
  train_size2 = len(roidb2)

  print('{:d} target roidb entries'.format(len(roidb2)))

  sampler_batch2 = sampler(train_size2, args.batch_size)

  dataset2 = roibatchLoader(roidb2, ratio_list2, ratio_index2, args.batch_size, \
                           imdb2.num_classes, training=True)

  dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=args.batch_size,
                            sampler=sampler_batch2, num_workers=args.num_workers)


  # guide1
  imdb3, roidb3, ratio_list3, ratio_index3 = combined_roidb(args.imdbguide_name)
  train_size3 = len(roidb3)

  print('{:d} distorted roidb entries'.format(len(roidb3)))

  sampler_batch3 = sampler(train_size3, args.batch_size)

  dataset3 = roibatchLoader(roidb3, ratio_list3, ratio_index3, args.batch_size, \
                            imdb3.num_classes, training=True)

  dataloader3 = torch.utils.data.DataLoader(dataset3, batch_size=args.batch_size,
                                            sampler=sampler_batch3, num_workers=args.num_workers)

  # guide2
  imdb4, roidb4, ratio_list4, ratio_index4 = combined_roidb(args.imdbguide2_name)
  train_size4 = len(roidb4)

  print('{:d} roidb entries'.format(len(roidb4)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  sampler_batch4 = sampler(train_size4, args.batch_size)

  dataset4 = roibatchLoader(roidb4, ratio_list4, ratio_index4, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader4 = torch.utils.data.DataLoader(dataset4, batch_size=args.batch_size,
                                           sampler=sampler_batch4, num_workers=args.num_workers)


# guide3
  imdb5, roidb5, ratio_list5, ratio_index5 = combined_roidb(args.imdbguide3_name)
  train_size5 = len(roidb5)

  print('{:d} roidb entries'.format(len(roidb5)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  sampler_batch5 = sampler(train_size5, args.batch_size)

  dataset5 = roibatchLoader(roidb5, ratio_list5, ratio_index5, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader5 = torch.utils.data.DataLoader(dataset5, batch_size=args.batch_size,
                                           sampler=sampler_batch5, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  dc_label = torch.FloatTensor(1)


  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    dc_label = dc_label.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)
  dc_label = Variable(dc_label)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr

  params = []
  params2 = []
  params3 = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'D_img' in key:
        if 'bias' in key:
            params += [{'params': [value], 'lr': lr *10 * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else :
          params += [{'params': [value], 'lr': lr*10 , 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

      else:
        if 'bias' in key:
          params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                      'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else:
          params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.cuda:
    fasterRCNN.cuda()

  iters_per_epoch = int(train_size / args.batch_size)
  iters_per_epoch2 = int(train_size2 / args.batch_size)
  first = 1
  optimizer.zero_grad()
  fasterRCNN.zero_grad()
  count = 0
  train_end = False
  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    data_iter = iter(dataloader)
    data_iter3 = iter(dataloader3)
    data_iter4 = iter(dataloader4)
    data_iter5 = iter(dataloader5)
    for step in range(iters_per_epoch):
      if (count+1) % 50000 == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma
      count += 1

      if step % iters_per_epoch2 == 0:
          data_target_iter = iter(dataloader2)
                       
      # SOURCE
      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
      dc_label_tmp =  torch.from_numpy(np.ones((2000, 1), dtype=np.float32))
      dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, RCNN_loss_img = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                             need_backprop=need_backprop, dc_label=dc_label)
      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
           + RCNN_loss_img.mean()
      loss.backward()

      # TARGET
      data2 = next(data_target_iter)
      im_data.data.resize_(data2[0].size()).copy_(data2[0])
      im_info.data.resize_(data2[1].size()).copy_(data2[1])
      gt_boxes.data.resize_(data2[2].size()).copy_(data2[2])
      num_boxes.data.resize_(data2[3].size()).copy_(data2[3])

      need_backprop = torch.from_numpy(np.zeros((1,), dtype=np.float32))
      dc_label_tmp =  torch.from_numpy(np.zeros((2000, 1), dtype=np.float32))
      dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

      rois2, cls_prob2, bbox_pred2, \
      rpn_loss_cls2, rpn_loss_box2, \
      RCNN_loss_cls2, RCNN_loss_bbox2, \
      rois_label2, RCNN_loss_img2 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                               need_backprop=need_backprop,  dc_label=dc_label)
      loss = RCNN_loss_img2.mean()
      loss.backward()


      # guide1
      data3 = next(data_iter3)
      im_data.data.resize_(data3[0].size()).copy_(data3[0])
      im_info.data.resize_(data3[1].size()).copy_(data3[1])
      gt_boxes.data.resize_(data3[2].size()).copy_(data3[2])
      num_boxes.data.resize_(data3[3].size()).copy_(data3[3])

      need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
      dc_label_tmp = torch.from_numpy(2 * np.ones((2000, 1), dtype=np.float32))
      dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

      rois3, cls_prob3, bbox_pred3, \
      rpn_loss_cls3, rpn_loss_box3, \
      RCNN_loss_cls3, RCNN_loss_bbox3, \
      rois_label3, RCNN_loss_img3 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                             need_backprop=need_backprop, dc_label=dc_label)
      loss = rpn_loss_cls3.mean() + rpn_loss_box3.mean() \
             + RCNN_loss_cls3.mean() + RCNN_loss_bbox3.mean() \
             + RCNN_loss_img3.mean()
      loss.backward()

      # guide2
      data4 = next(data_iter4)
      im_data.data.resize_(data4[0].size()).copy_(data4[0])
      im_info.data.resize_(data4[1].size()).copy_(data4[1])
      gt_boxes.data.resize_(data4[2].size()).copy_(data4[2])
      num_boxes.data.resize_(data4[3].size()).copy_(data4[3])

      need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
      dc_label_tmp = torch.from_numpy(3 * np.ones((2000, 1), dtype=np.float32))
      dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

      rois4, cls_prob4, bbox_pred4, \
      rpn_loss_cls4, rpn_loss_box4, \
      RCNN_loss_cls4, RCNN_loss_bbox4, \
      rois_label4, RCNN_loss_img4 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                             need_backprop=need_backprop, dc_label=dc_label)
      loss = rpn_loss_cls4.mean() + rpn_loss_box4.mean() \
             + RCNN_loss_cls4.mean() + RCNN_loss_bbox4.mean() \
             + RCNN_loss_img4.mean()  # + RCNN_loss_ins.mean()
      loss_temp += loss.data[0]
      loss.backward()
      
      # guide3
      data5 = next(data_iter5)
      im_data.data.resize_(data5[0].size()).copy_(data5[0])
      im_info.data.resize_(data5[1].size()).copy_(data5[1])
      gt_boxes.data.resize_(data5[2].size()).copy_(data5[2])
      num_boxes.data.resize_(data5[3].size()).copy_(data5[3])
      
      need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
      dc_label_tmp = torch.from_numpy(4 * np.ones((2000, 1), dtype=np.float32))
      dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)
      
      rois5, cls_prob5, bbox_pred5, \
      rpn_loss_cls5, rpn_loss_box5, \
      RCNN_loss_cls5, RCNN_loss_bbox5, \
      rois_label5, RCNN_loss_img5 = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                              need_backprop=need_backprop, dc_label=dc_label)
      loss = rpn_loss_cls5.mean() + rpn_loss_box5.mean() \
             + RCNN_loss_cls5.mean() + RCNN_loss_bbox5.mean() \
             + RCNN_loss_img5.mean()
      loss.backward()
      
      optimizer.step()
      optimizer.zero_grad()
      fasterRCNN.zero_grad()

      
      #f step % args.disp_interval == 0:
      # end = time.time()
      # start = time.time()

      if (count+1) % 10000 == 0:
        save_name = os.path.join(output_dir,
                               '{}_DDMRL3_trainval_{}_{}.pth'.format(args.dataset,
                                   args.session, count + 1))
        save_checkpoint({
          'session': args.session,
          # 'epoch': epoch + 1,
          'model': fasterRCNN.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    end = time.time()
    print(end - start)
