# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

import cPickle
import os

import cv2
import numpy as np
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.quad_transform import inclined_rect_transform_inv, clip_quads, polygon_area
import fastnms
from utils.blob import im_list_to_blob
from utils.timer import Timer


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        if cfg.TEST.MULTI_SCALES_NOC:
            blobs['im_info'] = np.hstack((
                np.array([[im_blob.shape[2], im_blob.shape[3]]], dtype=np.float32),
                im_scales.reshape((1, -1)).astype(np.float32)
            ))
        else:
            blobs['im_info'] = np.array(
                [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
                dtype=np.float32)
    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        if not cfg.TEST.MULTI_SCALES_NOC:
            assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        if cfg.TEST.MULTI_SCALES_NOC:
            assert rois.shape[0] % len(im_scales) == 0
            rois_per_scale = rois.shape[0] / len(im_scales)
            boxes = rois[:rois_per_scale, 1:5] / im_scales[0]
        else:
            boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    elif cfg.TEST.INCLINED_RECT_REG:
        box_deltas = blobs_out['inclined_rect_pred']
        pred_boxes = inclined_rect_transform_inv(boxes, box_deltas)
        pred_boxes = clip_quads(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in xrange(np.minimum(10, dets.shape[0])):
        if cfg.TEST.BBOX_REG or not cfg.TEST.USE_INCLINED_NMS:
            bbox = dets[i, :4]
            score = dets[i, -1]
            if score > thresh:
                # plt.cla()
                # plt.imshow(im)
                plt.gca().add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='red', linewidth=3)
                    )
        elif cfg.TEST.INCLINED_RECT_REG:
            bbox = dets[i, :8]
            score = dets[i, -1]
            if score > thresh:
                ax.add_patch(
                    plt.Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                 (bbox[4], bbox[5]), (bbox[6], bbox[7])], fill=False,
                                edgecolor='red', linewidth=3.5)
                )
    plt.title('{}  {:.3f}'.format(class_name, thresh))
    plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, max_per_image=400, thresh=-np.inf, vis=False, submit_dir='', submit_prefix=''):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    shapes = []
    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        print imdb.image_path_at(i)
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            if cfg.TEST.BBOX_REG:
                step = 4
            elif cfg.TEST.INCLINED_RECT_REG:
                step = 8
            else:
                print 'please config BBOX_REG or '
                return
            if cfg.TEST.AGNOSTIC:
                cls_boxes = boxes[inds, step:step*2]
            else:
                cls_boxes = boxes[inds, j*step:(j+1)*step]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            if cfg.TEST.BBOX_REG:
                assert not cfg.TEST.INCLINED_RECT_REG
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep, :]

            elif cfg.TEST.INCLINED_RECT_REG:
                assert not cfg.TEST.BBOX_REG
                if cfg.TEST.USE_INCLINED_NMS:
                    cls_dets = fastnms.standard_nms_n9(cls_dets.astype('float32'), cfg.TEST.NMS)
                else:
                    cls_dets_AABB = np.zeros((cls_dets.shape[0], 5), cls_dets.dtype)
                    # score
                    cls_dets_AABB[:, 4] = cls_dets[:, 8]
                    cls_dets_AABB[:, 0] = np.min(cls_dets[:, :8][:, 0::2], axis=1)
                    cls_dets_AABB[:, 1] = np.min(cls_dets[:, :8][:, 1::2], axis=1)
                    cls_dets_AABB[:, 2] = np.max(cls_dets[:, :8][:, 0::2], axis=1)
                    cls_dets_AABB[:, 3] = np.max(cls_dets[:, :8][:, 1::2], axis=1)
                    keep = nms(cls_dets_AABB, cfg.TEST.NMS)
                    cls_dets = cls_dets_AABB[keep, :]
                if vis:
                    vis_detections(im, imdb.classes[j], cls_dets, thresh=0.9)
            all_boxes[j][i] = cls_dets
            shapes.append(im.shape)

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    # det_file = os.path.join(output_dir, 'detections.pkl')
    # with open(det_file, 'wb') as f:
    #     cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    # added by hc
    write_boxes_file(imdb, all_boxes, shapes, submit_dir, submit_prefix)

    print 'Evaluating detections'
    # imdb.evaluate_detections(all_boxes, output_dir)

def write_boxes_file(imdb, all_boxes, shapes, submit_dir, submit_prefix):
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    # all_boxes already recovery to original image

    if not os.path.exists(submit_dir):
        os.makedirs(submit_dir)
    assert len(all_boxes) == 2
    num_images = len(all_boxes[1])
    for i in xrange(num_images):
        image_name = imdb.image_path_at(i).rsplit('/')[-1]
        shape = shapes[i]
        save_detection_path = submit_dir + '/' + submit_prefix + image_name[0:len(image_name)-4]+'.txt'
        cls_dets = all_boxes[1][i]
        detection_result = open(save_detection_path, 'wt')
        num_crds = cls_dets.shape[1]
        for j in xrange(len(cls_dets)):
            if num_crds == 5:
                xmin = int(np.round(cls_dets[j, 0]))
                ymin = int(np.round(cls_dets[j, 1]))
                xmax = int(np.round(cls_dets[j, 2]))
                ymax = int(np.round(cls_dets[j, 3]))
                conf = cls_dets[j, 4]

                xmin = max(1, xmin)
                ymin = max(1, ymin)
                xmax = min(shape[1] - 1, xmax)
                ymax = min(shape[0] - 1, ymax)
                result = str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymin) \
                         + ',' + str(xmax) + ',' + str(ymax) + ',' + str(xmin) + ',' + str(ymax) + ',' + str(
                    conf) + '\n'
                detection_result.write(result)
            elif num_crds == 9:
                x1 = int(np.round(cls_dets[j, 0]))
                y1 = int(np.round(cls_dets[j, 1]))
                x2 = int(np.round(cls_dets[j, 2]))
                y2 = int(np.round(cls_dets[j, 3]))
                x3 = int(np.round(cls_dets[j, 4]))
                y3 = int(np.round(cls_dets[j, 5]))
                x4 = int(np.round(cls_dets[j, 6]))
                y4 = int(np.round(cls_dets[j, 7]))
                conf = cls_dets[j, 8]

                x1 = max(1, x1)
                y1 = max(1, y1)
                x2 = min(shape[1] - 1, x2)
                y2 = max(1, y2)
                x3 = min(shape[1] - 1, x3)
                y3 = min(shape[0] - 1, y3)
                x4 = max(1, x4)
                y4 = min(shape[0] - 1, y4)
                # must clock-wise
                quad = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape(4, 2)
                if polygon_area(quad) >= 0:
                    print 'not clock-wise, image_name : %s, original crds:\n' % image_name
                    print quad.reshape((1, 8))
                    print 'only reverse direction, assuming that vertex is contious, after'
                    print quad[(0, 3, 2, 1), :]
                    result = str(x1) + ',' + str(y1) + ',' + str(x4) + ',' + str(y4) \
                             + ',' + str(x3) + ',' + str(y3) + ',' + str(x2) + ',' + str(y2) + ',' + str(conf) + '\n'
                else:
                    result = str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) \
                             + ',' + str(x3) + ',' + str(y3) + ',' + str(x4) + ',' + str(y4) + ',' + str(conf) + '\n'
                detection_result.write(result)
