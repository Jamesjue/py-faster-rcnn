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
