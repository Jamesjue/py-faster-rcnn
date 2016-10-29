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
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.dirname(__file__)
add_path(osp.join(this_dir, '..'))
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
from flask import Flask, request, redirect, url_for
from flask_restful import reqparse, abort, Api, Resource
import werkzeug
import json
import pdb
from werkzeug.utils import secure_filename
from tpod_detect import tpod_detect_image, init_net

CLASSES = ('__background__',
           'object')

app = Flask(__name__)
api = Api(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # set content max to be 16MB
app.config['UPLOAD_FOLDER'] = 'data'
UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
net=None

parser = reqparse.RequestParser()
parser.add_argument('picture', type=werkzeug.datastructures.FileStorage, location='files', required=True)
parser.add_argument('cf', type=float, location='minimum confidence score')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

querys={}
class Query(Resource):
    def get(self, query_id):
        if query_id in querys:
            return querys[query_id]
        else:
            abort(404, message="Query {} doesn't exist".format(query_id))
    
    def post(self, query_id):
        args = parser.parse_args()
        img = args['picture']
        min_cf = args['cf'] if 'cf' in args else 0.8
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
        detect_result=tpod_detect_image(net, bgr_img, min_cf)
        querys[query_id]=detect_result
        return detect_result, 201

api.add_resource(Query, '/<string:query_id>')

if __name__ == '__main__':
    global net
    base_dir='/py-faster-rcnn/models/tpod/VGG_CNN_M_1024/faster_rcnn_alt_op'
    prototxt = os.path.join(base_dir, 'faster_rcnn_test.py')
    caffemodel = os.path.join(base_dir, 'model.caffemodel')
    labelfile = os.path.join(base_dir, 'labels.txt')
    net=init_net()
    app.run(debug=True)
