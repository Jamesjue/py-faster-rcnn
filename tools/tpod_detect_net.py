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
try:
    import matplotlib as mpl

    mpl.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print 'no matplotlib. output bounding box in text only'
from flask import Blueprint, render_template, abort
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import sys
from textwrap import wrap
from tpod_utils import read_in_labels
import pdb
import os, fnmatch
import re

from flask import Flask
from flask import request, url_for, jsonify, Response, send_file

DEFAULT_CONFIDENCE = 0.6
PATH_RESULT = '/output.png'

app = Flask(__name__, static_url_path='/static', template_folder='/templates')


def get_latest_model_name():
    # init the net
    candidate_models = fnmatch.filter(os.listdir('.'), 'model_iter_*.caffemodel')
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


# init the net
caffemodel = get_latest_model_name()

prototxt = '/py-faster-rcnn/assembled_end2end/faster_rcnn_test.pt'
labelfile = '/train/labels.txt'
gpu_id = 0

assert os.path.exists(caffemodel), 'Path does not exist: {}'.format(caffemodel)

cfg.TEST.HAS_RPN = True  # Use RPN for proposals
caffe.set_mode_gpu()
caffe.set_device(gpu_id)
cfg.GPU_ID = gpu_id
net = caffe.Net(prototxt, caffemodel, caffe.TEST)
print '\n\nLoaded network {:s}'.format(caffemodel)
classes = read_in_labels(labelfile)


@app.route("/", methods=["GET"])
def visual_classifier():
    return render_template('index_visual.html')


@app.route('/detect', methods=["POST"])
def detect():
    # read the file
    print 'requested files %s ' % str(request.files)
    uploaded_files = []
    for k, v in request.files.items():
        uploaded_files.append(v)
    if len(uploaded_files) == 0:
        return Response('No file detected')
    print 'input images %s ' % str(uploaded_files)
    imgs =[]
    for img_file in uploaded_files:
        img_file.save(img_file.filename)
        print 'saved file %s ' % str(img_file.filename)
        img = cv2.imread(img_file.filename)
        imgs.append(img)

    confidence = float(DEFAULT_CONFIDENCE)
    # confidence can be None
    if 'confidence' in request.form:
        confidence = float(request.form['confidence'])
    else:
        print 'input confidence %s ' % str(confidence)
    # if ret_format is none, consider it 'box'
    ret_format = None
    if 'format' in request.form:
        ret_format = str(request.form['format'])

    global net
    # detect
    if len(imgs) == 1:
        # single image
        if ret_format is None or ret_format == 'box':
            ret = tpod_detect_image(net, imgs[0], classes, confidence)
            print 'detected result ' + str(ret)
            return Response(str(ret))
        else:
            dets = tpod_detect_image(net, imgs[0], classes, confidence)
            print 'detected result ' + str(dets)
            vis_detections(imgs[0], dets, confidence)
            plt.savefig(PATH_RESULT)
            return send_file(PATH_RESULT)
    else:
        # multiple images
        if ret_format is None or ret_format == 'box':
            ret = []
            for img in imgs:
                current_ret = tpod_detect_image(net, img, classes, confidence)
                print 'detected result ' + str(current_ret)
                ret.append(current_ret)
            return Response(str(ret))
        else:
            ret = None
            for i in range(0, len(imgs)):
                img = imgs[i]
                dets = tpod_detect_image(net, img, classes, confidence)
                print 'detected result ' + str(dets)
                vis_detections(img, dets, confidence)
                plt.savefig(PATH_RESULT)
                current_ret = cv2.imread(PATH_RESULT)
                if ret is None:
                    ret = current_ret
                else:
                    ret = np.vstack((ret, current_ret))
            cv2.imwrite(PATH_RESULT, ret)
            return send_file(PATH_RESULT)


def draw_bbox(ax, class_name, bbox, score):
    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linewidth=3.5)
    )
    ax.text(bbox[0], bbox[1] - 2,
            '{:s} {:.3f}'.format(class_name, score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=14, color='white')


def vis_detections(im, detect_rets, min_cf):
    fig, ax = plt.subplots(figsize=(12, 12))
    im_rgb = im[:, :, (2, 1, 0)]
    ax.imshow(im_rgb, aspect='equal')

    for (cls, bbox, score) in detect_rets:
        draw_bbox(ax, cls, bbox, score)

    ax.set_title('detected conf >= %s' % str(min_cf))
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def tpod_detect_image(net, im, classes, min_cf=0.8):
    """Detect object classes in an image using pre-computed object proposals."""
    # Detect all object classes and regress object bounds
    caffe.set_mode_gpu()
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    print 'returning only bx cf > {}'.format(min_cf)

    NMS_THRESH = 0.3
    ret = []
    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= min_cf)[0]
        for i in inds:
            bbox = map(float, list(dets[i, :4]))
            score = float(dets[i, -1])
            print 'detected {} at {} score:{}'.format(cls, bbox, score)
            ret.append((cls, bbox, score))
    return ret


def init_net(prototxt, caffemodel, labelfile, cfg, gpu_id):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    cfg.GPU_ID = gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    classes = read_in_labels(labelfile)
    return net, tuple(classes)

# def parse_args():
#     #    """Parse input arguments."""
#     #    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
#     #    parser.add_argument('im', help="Input image", default= '000456.jpg')
#     #    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
#     #                        default=0, type=int)
#     #    parser.add_argument('--cpu', dest='cpu_mode',
#     #                        help='Use CPU mode (overrides --gpu)',
#     #                        action='store_true')
#     #    parser.add_argument('--prototxt', dest='prototxt', help='Prototxt of Network')
#     #    parser.add_argument('--weights', dest='caffemodel', help='Weights of trained network')
#     #    parser.add_argument('--labels', dest='labels', help='file contain labels',
#     #                        default=None)
#     #    parser.add_argument('--cf', dest='min_cf', help='cutoff confidence score',
#     #                        default=0.8, type=float)
#     #    parser.add_argument('--output',
#     #                        dest='destination',
#     #                        help='Output location of image detections',
#     #                        default=None
#     #    )
#     #    args = parser.parse_args()
#
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(description='Faster R-CNN demo')
#     parser.add_argument('--input_image', dest='input_image', help="Input image", default='000456.jpg')
#     parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
#                         default=0, type=int)
#     parser.add_argument('--weights', dest='caffemodel', help='Weights of trained network')
#     parser.add_argument('--cf', dest='min_cf', help='cutoff confidence score',
#                         default=0.8, type=float)
#     parser.add_argument('--output',
#                         dest='destination',
#                         help='Output location of image detections',
#                         default=None
#                         )
#     args = parser.parse_args()
#
#     return args



# if __name__ == '__main__':
#     args = parse_args()
#     caffemodel = args.caffemodel
#
#     prototxt = '/py-faster-rcnn/assembled_end2end/faster_rcnn_test.pt'
#     labelfile = '/train/labels.txt'
#     gpu_id = args.gpu_id
#     input_path = args.input_image
#
#     assert os.path.exists(caffemodel), 'Path does not exist: {}'.format(caffemodel)
#     assert os.path.exists(input_path), 'Path does not exist: {}'.format(input_path)
#
#     net, classes = init_net(prototxt, caffemodel, labelfile, cfg, gpu_id)
#     im = cv2.imread(input_path)
#     dets = tpod_detect_image(net, im, classes, min_cf=args.min_cf)
#
#     if args.destination is not None and 'matplotlib' in sys.modules:
#         vis_detections(im, dets, args.min_cf)
#         plt.savefig(args.destination)
