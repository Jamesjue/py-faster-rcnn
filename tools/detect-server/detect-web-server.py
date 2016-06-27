#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
REST Web server for object detection using py-faster-rcnn

See README.md for installation instructions before running.
"""
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
from flask import Flask, request, redirect, url_for
from flask_restful import reqparse, abort, Api, Resource
import werkzeug
import json
import pdb
from werkzeug.utils import secure_filename

CLASSES = ('__background__',
           'object')

app = Flask(__name__)
api = Api(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # set content max to be 16MB
app.config['UPLOAD_FOLDER'] = 'data'
UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

parser = reqparse.RequestParser()
parser.add_argument('picture', type=werkzeug.datastructures.FileStorage, location='files')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

querys={}
class Query(Resource):
    def get(self, query_id):
        if query_id in querys:
            return {query_id: querys[query_id]}
        else:
            abort(404, message="Query {} doesn't exist".format(query_id))            

    def post(self, query_id):
        args = parser.parse_args()
        img = args['picture']
        filename = werkzeug.secure_filename(img.filename)
        mimetype = img.content_type
        print mimetype
        ctt=np.fromstring(img.read(), dtype=np.uint8)
        bgr_img=cv2.imdecode(ctt, cv2.IMREAD_UNCHANGED)
        cv2.imwrite('received.jpg', bgr_img)
        print bgr_img
        if img:
            filename = secure_filename(img.filename)
            mimetype = img.content_type
            if not allowed_file(img.filename):
                print 'not allowed:{} '.format(img.filename)
        detect_result=detect(bgr_img)
        querys[query_id]=detect_result
        return query_id, 201

api.add_resource(Query, '/<string:query_id>')
    
def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.75
    print 'threashold: {}'.format(CONF_THRESH)
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
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        for i in inds:
            bbox = map(str, list(dets[i, :4]))
            score = str(dets[i, -1])
            print 'detected roi:{} score:{}'.format(bbox, score)
            ret.append( (cls, bbox, score) )
    return ret

def detect(im, gpu=True, gpu_id=0):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    prototxt = os.path.join(cfg.MODELS_DIR, 'web_demo', 'VGG_CNN_M_1024',
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.MODELS_DIR, 'web_demo', 'VGG_CNN_M_1024',
                            'faster_rcnn_alt_opt', 'model.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if gpu:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        cfg.GPU_ID = gpu_id
    else:
        caffe.set_mode_cpu()        
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    return demo(net, im)

if __name__ == '__main__':
    app.run(debug=True)
