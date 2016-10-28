# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

#import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
import pdb

def parse_rec(filename):
    """ Parse a tpod annotation txt file """
    objects = []    
    with open(filename, 'r') as f:
        annos=f.read().splitlines()
        for each_anno in annos:
            items=each_anno.strip().split(' ')
            obj_struct = {}            
            if len(items) > 4:
                # newer version of annotation, with last item as cls
                obj_struct['name'] = items[-1]
            else:
                # older version of annotation, single class model, no cls name
                obj_struct['name'] = ''
                
            # lable all as not difficult
            # obj_struct bbox is using xtl, ytl, xbr, ybr
            # however, tpod stores bx as xtl,ytl,w,h
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [int(items[0]),
                                  int(items[1]),
                                  int(items[0])+int(items[2]),
                                  int(items[1])+int(items[3])]
            objects.append(obj_struct)
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# sample detpath.format(classname) detection file (text file)  # format: image_id, confidence, xtl, ytl, xbr, ybr, label
# 1 1.000 132.3 94.6 457.8 339.2
# 2 1.000 111.0 81.0 433.7 326.0
# 3 1.000 82.4 75.8 432.2 307.1
# 4 1.000 94.9 66.0 419.2 295.5

# sample annopath.format(image_id) annotation file (text file) # format: xtl, ytl, w, h, label
# 174 13 305 270 headphone

# sample imagesetfile:
# 1 /home/junjuew/object-detection-web/demo-web/vatic/videos/headphone3.mp4/0/0/0.jpg
# 2 /home/junjuew/object-detection-web/demo-web/vatic/videos/headphone3.mp4/0/0/1.jpg
# 3 /home/junjuew/object-detection-web/demo-web/vatic/videos/headphone3.mp4/0/0/2.jpg
# 4 /home/junjuew/object-detection-web/demo-web/vatic/videos/headphone3.mp4/0/0/3.jpg
# 5 /home/junjuew/object-detection-web/demo-web/vatic/videos/headphone3.mp4/0/0/4.jpg
# 6 /home/junjuew/object-detection-web/demo-web/vatic/videos/headphone3.mp4/0/0/5.jpg
    
def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the test annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    cachefile=None
    if cachedir != None:
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')
    
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip().split(' ')[0] for x in lines]

    if cachefile and os.path.isfile(cachefile):
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)
    else:
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        if cachefile:
            # save
            print 'Saving cached annotations to {:s}'.format(cachefile)
            with open(cachefile, 'w') as f:
                cPickle.dump(recs, f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        # TODO: for backward compatibility only. need a solution for it
        # 'object'==classname is there for models that doesn't have label.txt in their
        # train dir
        R = [obj for obj in recs[imagename] if obj['name'] == classname or obj['name'] == '' or 'object'==classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
