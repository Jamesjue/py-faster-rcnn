# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os, time
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from tpod_eval import voc_eval
from fast_rcnn.config import cfg
import pdb

'''
Documentation:
this class is used as the loader of the data set
it should accept one paths: the path for the training data folder, under this folder
there are three important files

1. (image_set.txt) the image set (the file containing the list of all image paths)
2. (label_set.txt) the label (the file containing the list of all label contents)
3. (labels.txt) the file containing name of all labels

On initialization, we can load all image paths and all labels into array, thus during the
training phase, we only need to read the array

'''

class tpod(imdb):
    TPOD_IMDB_NAME='web_demo'

    def __init__(self, devkit_path):
        # image_set should be a filename (without extension) in devkit_path that contains# a list of image paths
        # devkit_path should have an annotation dir, a label text
        imdb.__init__(self, self.__class__.TPOD_IMDB_NAME)
        self._year = '2016'

        if not devkit_path or not os.path.isdir(devkit_path):
            raise ValueError('Please provide image_set and devkit_path for tpod imdb. The devkit_path should contain '
                             'a text file with all labels named "label.txt". devkit_path/image_set.txt should be '
                             'a text file that each line is an absolute path to an image\n '
                             'Current devkit_path: {}'.format(devkit_path))

        self._image_set = 'train'
        self._devkit_path = devkit_path

        self._label_path = os.path.join(self._devkit_path, 'labels.txt')
        self._label_set_path = os.path.join(self._devkit_path, 'label_set.txt')
        self._image_set_path = os.path.join(self._devkit_path, 'image_set.txt')

        # load classes names
        if os.path.isfile(self._label_path):
            self._classes = self.load_label_classes()
        else:
            print 'no label file given, model is single class'
            # backward compatibility, single class
            self._classes = ('__background__', # always index 0
                             'object')

        self._image_index = self._load_image_set_index()
        self._annotation_index, self._obj_num_index = self._load_annotations()

        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = time.strftime("%Y%m%d%H%M%S")

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}
        print 'initialized imdb from devkit_path: {}, image_set: {}'.format(self._devkit_path, self._image_set)

    def load_label_classes(self):
        f = open(self._label_path, 'r')
        labels = f.read().splitlines()
        labels.insert(0, '__background__')
        print 'custom labels: {}'.format(labels)
        return labels

    # need customization
    def _is_test(self):
        return False

    # need customization
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        # Example path to image set file:
        # self._devkit_path + /image_set.txt
        """
        f = open(self._image_set_path, 'r')
        # at the same time remove the empty space at the beginning and end
        image_paths = [x.strip().split() for x in f.readlines()]
        return image_paths

    def _load_annotations(self):
        # the basic structure: class is separated by '.' label is separated by ';' coordination is separated by ','
        f = open(self._label_set_path, 'r')
        lines = f.readlines()
        annotation_set = []
        object_num_set = []
        for line in lines:
            obj_num = 0
            line_classes = line.split('.')
            line_annotation = []
            for i, line_class in enumerate(line_classes):
                # there might be extra separation symbol,
                # we should ignore these more than actual classes
                if i >= len(self._classes) - 1:
                    break
                line_label = []
                if len(line_class) > 1:
                    labels = line_class.split(';')
                    for label in labels:
                        if len(label) < 1:
                            continue
                        coordination = label.split(',')
                        line_label.append(coordination)
                        obj_num += 1
                line_annotation.append(line_label)
            annotation_set.append(line_annotation)
            object_num_set.append(obj_num)
        return annotation_set, object_num_set

    # need customization
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        image_path = '/' + self._image_index[i][-1]
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # need customization
    def _load_tpod_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        index = int(index)
        frame_label = self._annotation_index[index]
        # we should include the background inside
        num_objs = self._obj_num_index[index] + 1

        print 'load tpod annotation %s num objs %s ' % (str(index), str(num_objs))

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # n: the number of objects
        # c: the number of classes
        # boxes: a n x 4 matrix, the rect for each object, each row is a box, n is the number of objects
        # gt_classes: ground truth, a n length array, each element is the class index the object
        # overlaps: a n x c matrix, in each row, if that object appears, set it to be 1
        # seg_areas: a n length array, each element is the area size for that object

        # Load object bounding boxes into a data frame.

        idx = 0
        for i, current_class in enumerate(frame_label):
            if len(current_class) > 0 and i < self.num_classes:
                for label in current_class:
                    if len(label) > 0:
                        x1 = float(label[0])
                        y1 = float(label[1])
                        w = float(label[2])
                        h = float(label[3])

                        x2 = x1 + w
                        y2 = y1 + h
                        # j: class is the last word in an entry separated by white space
                        cls = i + 1
                        boxes[idx, :] = [x1, y1, x2, y2]
                        gt_classes[idx] = cls
                        overlaps[idx, cls] = 1.0
                        seg_areas[idx] = w * h

                        idx += 1

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    @property
    def cache_path(self):
        return self._devkit_path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_tpod_annotation(index)
                    for index, path in enumerate(self.image_index)]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if not self._is_test():
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        print 'rpn_roidb called'
        if not self._is_test():
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print '_load_rpn_roidb called'
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _get_comp_id(self):
        return self._comp_id

    def _get_voc_results_file_template(self, prefix):
        filename = self._get_comp_id() + '_det_' + self._image_set.replace('/', '_') + '_{:s}.txt'
        path = os.path.join(
            prefix,
            filename)
        return path

    def _write_voc_results_file(self, all_boxes, output_dir):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template(output_dir).format(cls)
            print 'Writing {} VOC results file ==> {}'.format(cls, filename)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[0], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    # need customization
    def _do_python_eval(self, annopath, output_dir, cachedir=None):
        '''annopath = '/home/junjuew/object-detection-web/demo-web/train/headphone-model/Annotations/{}.txt'
        '''
        imagesetfile = os.path.join(
            self._devkit_path,
            self._image_set + '.txt')

        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template(output_dir).format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            print('precision for {} = {}'.format(cls, prec))
            print('recall for {} = {}'.format(cls, rec))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, annopath, output_dir):
        self._write_voc_results_file(all_boxes, output_dir)
        self._do_python_eval(annopath, output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template(output_dir).format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.tpod import tpod
    d = tpod('trainval', '')
    res = d.roidb
    from IPython import embed; embed()

