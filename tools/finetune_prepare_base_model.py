#!/usr/bin/env python
"""Prepare base model to finetune from.

The base model should have different layer names in the end.
"""
import _init_paths
import caffe
import os
import fire


def print_caffemodel_params(model_path, model_prototxt_path):
    net = caffe.Net(model_prototxt_path,
                    model_path,
                    caffe.TEST)
    for layername, layerparam in net.params.items():
        print '  Layer Name : {0:>7}, Weight Dims :{1:12} '.format(
            layername, layerparam[0].data.shape)
        print '--------------------------------------------------------'
    import pdb
    pdb.set_trace()


def rename_final_layers(model_path):
    """Used to create a base model weight file for finetuning.

    Without renaming final layers, finetuning requires changing the names of
    output layers. However, py-faster-rcnn code(lib/train.py, lib/test.py, etc)
    hardcoded final final layers 'cls_score', 'bbox_pred' into their
    code. Therefore, finetuning requires chaning these codes as well.

    An easier method is just to rename the layer first. And use the weight file
    with the renamed layer names, such that the 'cls_score', 'bbox_pred' layers
    in finetuning prototxt can be re-intialized.

    """
    renamed_net = caffe.Net(
        'vgg16_templates/renamed_output_layers_test.prototxt',
        model_path,
        caffe.TEST)
    renamed_net.save('VGG16_faster_rcnn_finetune.caffemodel')


if __name__ == '__main__':
    fire.Fire()
