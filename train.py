# --------------------------------------------------------
# Pytorch Diversify and Match
# Licensed under The MIT License [see LICENSE for details]
# Written by Taeykyung Kim based on codes from Jiasen Lu, Jianwei Yang, and Ross Girshick
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
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

from roi_da_data_layer.create_loader import create_dataloader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, clip_gradient
from model.faster_rcnn.DivMatch_vgg16 import vgg16
# from model.faster_rcnn.ImgMultiGRL_resnet import resnet

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--steps', type=int, default=80000, metavar='N',
                        help='maximum number of iterations '
                             'to train (default: 80000)')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default="pascal_voc", type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='res101', type=str)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=10, type=int)
    parser.add_argument('--save_interval', dest='save_interval',
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
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=10022, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=False, type=bool)

    args = parser.parse_args()
    return args

def input2loss(data, need_backprop, dc_label, step, disp_interval):
    im_data.data.resize_(data[0].size()).copy_(data[0])
    im_info.data.resize_(data[1].size()).copy_(data[1])
    gt_boxes.data.resize_(data[2].size()).copy_(data[2])
    num_boxes.data.resize_(data[3].size()).copy_(data[3])

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, DA_loss_dom = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,
                                             need_backprop=need_backprop, dc_label=dc_label)

    if need_backprop.numpy():
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() \
               + DA_loss_dom.mean()

        if (step + 1) % disp_interval == 0:
            print('rpn_cls: ', '%0.5f' % rpn_loss_cls.cpu().data.numpy(),
                  ' | rpn_box: ', '%0.5f' % rpn_loss_box.cpu().data.numpy(),
                  ' | RCNN_cls: ', '%0.5f' % RCNN_loss_cls.cpu().data.numpy(),
                  ' | RCNN_bbox: ', '%0.5f' % RCNN_loss_bbox.cpu().data.numpy(),
                  ' | DA_dom: ', '%0.5f' % DA_loss_dom.cpu().data.numpy()
                  )
        return loss, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, DA_loss_dom
    else:
        loss = DA_loss_dom.mean()

        if (step + 1) % disp_interval == 0:
            print('rpn_cls: ', '%0.5f' % 0,
                  ' | rpn_box: ', '%0.5f' % 0,
                  ' | RCNN_cls: ', '%0.5f' % 0,
                  ' | RCNN_bbox: ', '%0.5f' % 0,
                  ' | DA_dom: ', '%0.5f' % DA_loss_dom.cpu().data.numpy()
            )
        return loss, DA_loss_dom

if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)

    if args.use_tfboard:
        from model.utils.logger import Logger
        # Set the logger
        logger = Logger('./logs')

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.dataset in ["clipart", "watercolor", "comic"]:
        print(args.dataset)
        args.imdb_name = "voc_integrated_trainval"
        args.imdbval_name = args.dataset + "_trainval"
        args.imdb_shifted1_name = args.dataset + "CP_trainval"
        args.imdb_shifted2_name = args.dataset + "R_trainval"
        args.imdb_shifted3_name = args.dataset + "CPR_trainval"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    elif args.dataset == "cityscapes":
        args.imdb_name = "cityscapes_train"
        args.imdbval_name = "foggy_cityscapes_val"
        args.imdbguide_name = "CityscapesCP_train"
        args.imdbguide2_name = "CityscapesR_train"
        args.imdbguide3_name = "CityscapesCPR_train"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ####################################################################################################################
    ################################################## load train set ##################################################
    ####################################################################################################################
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    print(args.imdb_name, args.imdbval_name, args.imdb_shifted1_name)
    print('---------------------------------------------------------------------------------')
    dataloader, train_size, imdb = create_dataloader(args.imdb_name, args)
    print('---------------------------------------------------------------------------------')
    dataloader2, train_size2, _ = create_dataloader(args.imdbval_name, args)
    print('---------------------------------------------------------------------------------')
    dataloader3, train_size3, _ = create_dataloader(args.imdb_shifted1_name, args)
    print('---------------------------------------------------------------------------------')
    dataloader4, train_size4, _ = create_dataloader(args.imdb_shifted2_name, args)
    print('---------------------------------------------------------------------------------')
    dataloader5, train_size5, _ = create_dataloader(args.imdb_shifted3_name, args)
    print('---------------------------------------------------------------------------------')

    ####################################################################################################################
    ########################################### initialize the tensor holder ###########################################
    ####################################################################################################################
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

    ####################################################################################################################
    ################################################### load network ###################################################
    ####################################################################################################################
    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    # elif args.net == 'res50':
    #     fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    # elif args.net == 'res101':
    #     fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    # elif args.net == 'res152':
    #     fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'Dis' in key:
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
    optimizer.zero_grad()
    fasterRCNN.zero_grad()

    iters_per_epoch = int(train_size / args.batch_size)
    iters_per_epoch2 = int(train_size2 / args.batch_size)
    first = 1
    count = 0
    train_end = False
    for step in range(args.steps):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if step % iters_per_epoch == 0:
            data_iter = iter(dataloader)
            data_iter3 = iter(dataloader3)
            data_iter4 = iter(dataloader4)
            data_iter5 = iter(dataloader5)

        if step % iters_per_epoch2 == 0:
            data_target_iter = iter(dataloader2)

        if (step + 1) % 50000 == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma
        if (step + 1) % args.disp_interval == 0:
            print('\n', '[{} iters  / {} iters]'.format(step, args.steps))
        # SOURCE
        data = next(data_iter)
        need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
        dc_label_tmp = torch.from_numpy(np.ones((2000, 1), dtype=np.float32))
        dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

        loss, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, DA_loss_dom = input2loss(data, need_backprop, dc_label, step, args.disp_interval)
        loss.backward()

        # TARGET
        data2 = next(data_target_iter)
        need_backprop = torch.from_numpy(np.zeros((1,), dtype=np.float32))
        dc_label_tmp = torch.from_numpy(np.zeros((2000, 1), dtype=np.float32))
        dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

        loss, DA_loss_dom = input2loss(data, need_backprop, dc_label, step, args.disp_interval)
        loss.backward()

        # guide1
        data3 = next(data_iter3)
        need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
        dc_label_tmp = torch.from_numpy(2 * np.ones((2000, 1), dtype=np.float32))
        dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

        loss, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, DA_loss_dom = input2loss(data3, need_backprop, dc_label, step, args.disp_interval)
        loss.backward()

        # guide2
        data4 = next(data_iter4)
        need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
        dc_label_tmp = torch.from_numpy(3 * np.ones((2000, 1), dtype=np.float32))
        dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

        loss, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, DA_loss_dom = input2loss(data4, need_backprop, dc_label, step, args.disp_interval)
        loss.backward()

        # guide3
        data5 = next(data_iter5)
        need_backprop = torch.from_numpy(np.ones((1,), dtype=np.float32))
        dc_label_tmp = torch.from_numpy(4 * np.ones((2000, 1), dtype=np.float32))
        dc_label.data.resize_(dc_label_tmp.size()).copy_(dc_label_tmp)

        loss, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, DA_loss_dom = input2loss(data5, need_backprop, dc_label, step, args.disp_interval)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        fasterRCNN.zero_grad()

        if (step + 1) % args.save_interval == 0:
            save_name = os.path.join(output_dir,
                                   '{}_DivMatch_trainval_{}_{}.pth'.format(args.dataset, args.session, step + 1))
            save_checkpoint({
              'session': args.session,
              'model': fasterRCNN.state_dict(),
              'optimizer': optimizer.state_dict(),
              'pooling_mode': cfg.POOLING_MODE,
              'class_agnostic': args.class_agnostic,
            }, save_name)
            print('save model: {}'.format(save_name))

        # end = time.time()
        # print(end - start)
