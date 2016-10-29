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

CLASSES = ('__background__',
           'object')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

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

    ax.set_title('detected conf >= {:.2f}'.format(min_cf))
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    
def tpod_detect_image(net, im, min_cf=0.8):
    """Detect object classes in an image using pre-computed object proposals."""
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    print 'returning only bx cf > {}'.format(min_cf)
    
    NMS_THRESH = 0.3
    ret=[]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= min_cf)[0]
        for i in inds:
            bbox = list(dets[i, :4])
            score = dets[i, -1]
            print 'detected {} at {} score:{}'.format(cls, bbox, score)
            ret.append( (cls, bbox, score) )
    return ret

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('im', help="Input image", default= '000456.jpg')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--prototxt', dest='prototxt', help='Prototxt of Network')
    parser.add_argument('--weights', dest='caffemodel', help='Weights of trained network')
    parser.add_argument('--labels', dest='labels', help='file contain labels',
                        default=None)
    parser.add_argument('--cf', dest='min_cf', help='cutoff confidence score',
                        default=0.8, type=float) 
    parser.add_argument('--output',
                        dest='destination',
                        help='Output location of image detections',
                        default=None
    )
    args = parser.parse_args()

    return args

def init_net(prototxt, caffemodel, labelfile, cpu_mode=False):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    if cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)
    CLASSES=read_in_labels(labelfile)
    return net
    
if __name__ == '__main__':
    args = parse_args()
    prototxt = args.prototxt
    caffemodel = args.caffemodel
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))
    net=init_net(prototxt, caffemodel, args.labels, cpu_mode=False)
    im=cv2.imread(args.im)
    dets=tpod_detect_image(net, im, min_cf=args.min_cf)
    
    if args.destination is not None and 'matplotlib' in sys.modules:
        vis_detections(im, dets, args.min_cf)
        plt.savefig(args.destination)
    
    # Warmup on a dummy image
    # im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _= im_detect(net, im)


    #im_names = args.im#['000456.jpg', '000542.jpg', '001150.jpg',
                #'001763.jpg', '004545.jpg']
    #for im_name in im_names:
    #    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #    print 'Demo for data/demo/{}'.format(im_name)
    #    demo(net, im_name)

#    plt.show()



