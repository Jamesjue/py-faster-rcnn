#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
import matplotlib
matplotlib.use('Agg')
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import os
from datasets.tpod_dataset import tpod
import tpod_utils
import shutil
import os
import fnmatch
import re

TRAIN_PATH = '/train/'

'''
# arguments needed
-- gpu
-- devkit_path
-- output_dir
-- iters
-- weights
--

'''


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output directory', default=None,
                        type=str)
    parser.add_argument('--input_dir', dest='input_dir',
                        help=('the dir should have image_list,'
                              'label_list, and label_name three files.'
                              'See github wiki'),
                        default=None, type=str)
    parser.add_argument('--template_dir', dest='template_dir',
                        help=('Input caffe network prototxt template dir.'
                              'should have the correct notation '
                              'with $$ and %%'),
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(devkit_path):
    # we assume that there is only one imdb
    imdb = tpod(devkit_path)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    roidb = get_training_roidb(imdb)
    return imdb, roidb


def prepare_network_structures(num_objs, template_dir, output_dir):
    tpod_utils.prepare_prototxt_files(num_objs, template_dir, output_dir)


def get_labels(path):
    f = open(path, 'r')
    labels = f.read().splitlines()
    labels.insert(0, '__background__')
    return labels


def get_latest_model_name():
    # init the net
    candidate_models = fnmatch.filter(
        os.listdir('.'), 'model_iter_*.caffemodel')
    assert len(candidate_models) > 0, 'No model file detected'
    model = candidate_models[0]
    max_iteration = -1
    for candidate in candidate_models:
        iteration_match = re.search(r'model_iter_(\d+)\.caffemodel', candidate)
        if iteration_match:
            iteration = int(iteration_match.group(1))
            if max_iteration < iteration:
                max_iteration = iteration
                model = candidate
    return model


if __name__ == '__main__':
    args = parse_args()
    print '--- begin main of tpod train net.py'
    print 'Parameters: gpu %s, iters %s, weights %s, input_dir %s, template_dir %s, output_dir %s' %\
        (str(args.gpu_id), str(args.max_iters), str(args.pretrained_model),
         str(args.input_dir),
         str(args.template_dir),
         str(args.output_dir))

    # prepare the train data set
    train_image_list_path = os.path.join(args.input_dir, 'image_list.txt')
    train_label_list_path = os.path.join(args.input_dir, 'label_list.txt')
    train_label_name_path = os.path.join(args.input_dir, 'label_name.txt')
    assert os.path.exists(train_image_list_path), 'Path does not exist: {}'.format(
        train_image_list_path)
    assert os.path.exists(train_label_list_path), 'Path does not exist: {}'.format(
        train_label_list_path)
    assert os.path.exists(train_label_name_path), 'Path does not exist: {}'.format(
        train_label_name_path)

    # background included
    labels = get_labels(train_label_name_path)

    # first, prepare the network structure files, their paths
    '''
    faster_rcnn_test.pt
    solver.prototxt
    train.prototxt
    '''
    prepare_network_structures(len(labels), args.template_dir, args.output_dir)

    # read cfg file
    cfg_from_file(os.path.join(args.template_dir, 'faster_rcnn_end2end.yml'))

    # gpu
    cfg.GPU_ID = args.gpu_id
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    print('Using config:')
    pprint.pprint(cfg)

    # read paths
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    else:
        # clear the folder
        shutil.rmtree(TRAIN_PATH)
        os.makedirs(TRAIN_PATH)

    target_image_list_path = TRAIN_PATH + 'image_set.txt'
    target_label_list_path = TRAIN_PATH + 'label_set.txt'
    target_label_name_path = TRAIN_PATH + 'labels.txt'
    shutil.copyfile(train_image_list_path, target_image_list_path)
    shutil.copyfile(train_label_list_path, target_label_list_path)
    shutil.copyfile(train_label_name_path, target_label_name_path)

    # get data set
    imdb, roidb = combined_roidb(TRAIN_PATH)
    print '{:d} roidb entries'.format(len(roidb))

    solver_path = os.path.join(args.output_dir, 'solver.prototxt')
    # begin training
    train_net(solver_path, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
