#!/usr/bin/env python
import _init_paths
import numpy as np
import os

def read_in_labels(fp):
    if fp and os.path.isfile(fp):
        with open(fp, 'r') as f:
            labels=f.read().splitlines()
        labels.insert(0, '__background__')
        return labels
    else:
        raise IOError('read from labelfile {} failed!'.format(fp))


# the label_cnt should contains the background class
def prepare_prototxt_files(label_cnt, input_folder, output_folder):
    # there is always a __background__ class
    # input file name: output file name
    customize_files = {
        'solver.prototxt':'solver.prototxt',
        'train.prototxt':'train.prototxt',
        'test.prototxt':'faster_rcnn_test.pt'
    }
    for ef, efo in customize_files.iteritems():
        fpath = os.path.join(input_folder, ef)
        foutpath=os.path.join(output_folder, efo)
        with open(fpath, 'r') as f:
            content=f.read()
        # particular rule used for rcnn_alt_opt
        content=content.replace('%%', str(label_cnt))
        content=content.replace('##', str(4*label_cnt))
        content=content.replace('$$', os.path.join(str(output_folder), 'train.prototxt'))
        with open(foutpath, 'w') as f:
            f.write(content)

