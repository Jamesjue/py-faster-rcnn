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
from datasets.tpod import tpod

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

#    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
#    parser.add_argument('--gpu', dest='gpu_id',
#                        help='GPU device id to use [0]',
#                        default=0, type=int)
#    parser.add_argument('--solver', dest='solver',
#                        help='solver prototxt',
#                        default=None, type=str)
#    parser.add_argument('--image_set', dest='image_set',
#                        help='a file in which each line is a file path',
#                        default=None, type=str)
#    parser.add_argument('--devkit_path', dest='devkit_path',
#                        help='devkit path',
#                        default=None, type=str)
#    parser.add_argument('--iters', dest='max_iters',
#                        help='number of iterations to train',
#                        default=40000, type=int)
#    parser.add_argument('--weights', dest='pretrained_model',
#                        help='initialize with pretrained model weights',
#                        default=None, type=str)
#    parser.add_argument('--cfg', dest='cfg_file',
#                        help='optional config file',
#                        default=None, type=str)
#    parser.add_argument('--rand', dest='randomize',
#                        help='randomize (do not use a fixed seed)',
#                        action='store_true')
#    parser.add_argument('--set', dest='set_cfgs',
#                        help='set config keys', default=None,
#                        nargs=argparse.REMAINDER)
#    parser.add_argument('--output_dir', dest='output_dir',
#                        help='output directory', default=None,
#                        type=str)
#    parser.add_argument('--final',
#                        dest='final_path',
#                        help='output path',
#                        default=None)

    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--devkit_path', dest='devkit_path',
                        help='devkit path',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output directory', default=None,
                        type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(imdb_names, image_set, devkit_path):
    def get_roidb(imdb_name, image_set, devkit_path):
        imdb = get_imdb(imdb_name, image_set, devkit_path)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s, image_set=image_set, devkit_path=devkit_path)
              for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        print('multiple imdb')        
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        print('single imdb')
        imdb = get_imdb(imdb_names, image_set, devkit_path)
    return imdb, roidb

if __name__ == '__main__':
    args = parse_args()
    print '--- begin main of tpod train net.py'
    print 'command line parameters: gpu %s, dev_path %s, iters %s, weights %s, output_dir %s' %\
    (str(args.gpu_id), str(args.devkit_path), str(args.max_iters), str(args.pretrained_model), str(args.output_dir))


'''
def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000):
'''

'''
Necessary parameters
cfg file
gpu
imdb, roidb
output_dir

'''



#    args.imdb_name=tpod.TPOD_IMDB_NAME
#
#    if args.cfg_file is not None:
#        cfg_from_file(args.cfg_file)
#    if args.set_cfgs is not None:
#        cfg_from_list(args.set_cfgs)
#
#    assert (args.image_set is not None)
#    assert (args.devkit_path is not None)
#    assert (args.output_dir is not None)
#    args.image_set = os.path.abspath(args.image_set)
#    args.devkit_path = os.path.abspath(args.devkit_path)
#    args.output_dir = os.path.abspath(args.output_dir)
#
#    print('Called with args:')
#    print(args)
#
#    cfg.GPU_ID = args.gpu_id
#
#    print('Using config:')
#    pprint.pprint(cfg)
#
#    if not args.randomize:
#        # fix the random seeds (numpy and caffe) for reproducibility
#        np.random.seed(cfg.RNG_SEED)
#        caffe.set_random_seed(cfg.RNG_SEED)
#
#    # set up caffe
#    caffe.set_mode_gpu()
#    caffe.set_device(args.gpu_id)
#
#    imdb, roidb = combined_roidb(args.imdb_name, args.image_set, args.devkit_path)
#    print '{:d} roidb entries'.format(len(roidb))
#
#    output_dir = args.output_dir
#    print 'Output will be saved to `{:s}`'.format(output_dir)
#
#    train_net(args.solver, roidb, output_dir,
#              pretrained_model=args.pretrained_model,
#              max_iters=args.max_iters)
