import os, time
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import subprocess
import uuid

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

class tpod():
    TPOD_IMDB_NAME='web_demo'

    def __init__(self, image_set, devkit_path):
        # image_set should be a filename (without extension) in devkit_path that contains# a list of image paths
        # devkit_path should have an annotation dir, a label text
        self._year = '2016'

        if not image_set or not devkit_path or not os.path.isdir(devkit_path):
            raise ValueError('Please provide image_set and devkit_path for tpod imdb. The devkit_path should contain '
                             'a text file with all labels named "label.txt". devkit_path/image_set.txt should be '
                             'a text file that each line is an absolute path to an image\n '
                             'Current image_set: {} devkit_path: {}'.format(image_set, devkit_path))

        self._image_set = image_set
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
        # not sure whether to minus 1
        self.num_classes = len(self._classes) - 1

        self._image_index = self._load_image_set_index()
        self._annotation_index, self._obj_num_index = self._load_annotations()

        print self._image_index

        print 'initialized imdb from devkit_path: {}, image_set: {}'.format(self._devkit_path, self._image_set)

    def load_label_classes(self):
        f = open(self._label_path, 'r')
        labels = f.read().splitlines()
        labels.insert(0, '__background__')
        print 'custom labels: {}'.format(labels)
        return labels

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
        image_path = self._image_index[i][-1]
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # need customization
    def _load_tpod_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        frame_label = self._annotation_index[index]
        num_objs = self._obj_num_index[index]

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
                        cls = i
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

