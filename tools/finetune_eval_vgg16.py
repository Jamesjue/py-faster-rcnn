#!/usr/bin/env python

"""Test a Fast R-CNN network on an image database."""

import _init_paths
# from fast_rcnn.test import test_net
import matplotlib

matplotlib.use('Agg')
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect
from datasets.factory import get_imdb
from datasets.tpod_dataset import tpod
import caffe
import pprint
import time
import os
import sys
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import pdb
import shutil
import tpod_utils
import os
import fnmatch
import re

WORKING_PATH = '/work/'


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Evaluate a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='output directory', default=None,
                        type=str)
    parser.add_argument('--input_dir', dest='input_dir',
                        help=('the dir should have image_list,'
                              'label_list, and label_name three files.'
                              'See github wiki'),
                        default=None, type=str)
    parser.add_argument('--model_dir', dest='model_dir',
                        help=('caffe network prototxt dir.'
                              'Trained model weights should exist.'
                              'cfg_file should also exist.'),
                        default=None, type=str)
    parser.add_argument('--eval_result_name', dest='eval_result_name',
                        help='the name of the evaluation result folder')
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def eval_net(net, imdb, label_list_path, evaluation_result_name, output_dir, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    assert cfg.TEST.HAS_RPN, 'HAS_RPN not asserted in cfg'

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    det_file = os.path.join(output_dir, 'detections.pkl')
    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
            .format(i + 1, num_images, _t['im_detect'].average_time,
                    _t['misc'].average_time)

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, label_list_path,
                             output_dir, evaluation_result_name)


def prepare_network_structures(num_objs, template_dir, output_dir):
    tpod_utils.prepare_prototxt_files(num_objs, template_dir, output_dir)


def get_labels(path):
    f = open(path, 'r')
    labels = f.read().splitlines()
    labels.insert(0, '__background__')
    return labels


def get_latest_model_name(model_dir):
    # init the net
    candidate_models = fnmatch.filter(
        os.listdir(model_dir), '*_iter_*.caffemodel')
    assert len(candidate_models) > 0, 'No model file detected'
    model = candidate_models[0]
    max_iteration = -1
    for candidate in candidate_models:
        iteration_match = re.search(r'.*_iter_(\d+)\.caffemodel', candidate)
        if iteration_match:
            iteration = int(iteration_match.group(1))
            if max_iteration < iteration:
                max_iteration = iteration
                model = candidate
    return model


if __name__ == '__main__':
    args = parse_args()
    print '--- begin main of tpod train net.py'
    print 'Parameters: gpu %s, input_dir %s, model_dir %s, output_dir %s' % \
        (str(args.gpu_id),
         str(args.input_dir),
         str(args.model_dir),
         str(args.output_dir))

    evaluation_result_name = args.eval_result_name

    # check evaluation set
    eval_image_list_path = os.path.join(args.input_dir, 'image_list.txt')
    eval_label_list_path = os.path.join(args.input_dir, 'label_list.txt')
    eval_label_name_path = os.path.join(args.input_dir, 'label_name.txt')
    assert os.path.exists(eval_image_list_path), 'Path does not exist: {}'.format(
        eval_image_list_path)
    assert os.path.exists(eval_label_list_path), 'Path does not exist: {}'.format(
        eval_label_list_path)
    assert os.path.exists(eval_label_name_path), 'Path does not exist: {}'.format(
        eval_label_name_path)

    os.makedirs(args.output_dir)
    
    # background included
    labels = get_labels(eval_label_name_path)

    # first, prepare the network structure files, their paths
    '''
    faster_rcnn_test.pt
    solver.prototxt
    train.prototxt
    '''
    # prepare_network_structures(len(labels), args.model_dir, args.output_dir)

    # read cfg file
    cfg_from_file(os.path.join(args.model_dir, 'faster_rcnn_end2end.yml'))

    # gpu
    cfg.GPU_ID = args.gpu_id
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # TODO: turn on flag to remove intermediate files
    cfg.cleanup = False  # keep intermediate detection

    print('Using config:')
    pprint.pprint(cfg)

    latest_model = os.path.join(args.model_dir, get_latest_model_name(args.model_dir))
    print 'get latest model: %s ' % str(latest_model)

    prototxt = os.path.join(args.model_dir, 'faster_rcnn_test.pt')

    net = caffe.Net(prototxt, latest_model, caffe.TEST)

    # prepare data set
    if not os.path.exists(WORKING_PATH):
        os.makedirs(WORKING_PATH)
    else:
        # clear the folder
        shutil.rmtree(WORKING_PATH)
        os.makedirs(WORKING_PATH)

    target_image_list_path = WORKING_PATH + 'image_set.txt'
    target_label_list_path = WORKING_PATH + 'label_set.txt'
    target_label_name_path = WORKING_PATH + 'labels.txt'
    shutil.copyfile(eval_image_list_path, target_image_list_path)
    shutil.copyfile(eval_label_list_path, target_label_list_path)
    shutil.copyfile(eval_label_name_path, target_label_name_path)

    imdb = tpod(WORKING_PATH)
    print 'Loaded dataset `{:s}` for evaluation'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    imdb.competition_mode(args.comp_mode)

    # begin evaluation
    eval_net(net, imdb, eval_label_list_path,
             evaluation_result_name, args.output_dir, 100)
