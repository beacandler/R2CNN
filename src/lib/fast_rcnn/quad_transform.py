#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""some transform of the gt quadrilaterals"""

import numpy as np
import cv2
import math
from fast_rcnn.icdar import *
def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.

def find_order_boxes(inclined_rect, quad):
    """map first vertex in quad to first vertex in inclined rect according distance"""
    if inclined_rect.shape == (4, 1, 2):
        inclined_rect = inclined_rect.reshape((4, 2))

    if polygon_area(inclined_rect) > 0:# must clock-wise
        inclined_rect = inclined_rect[(0, 3, 2, 1), :]
        assert polygon_area(inclined_rect) < 0

    first_ind = -1
    min_dis =  np.inf
    for i in xrange(4):
        dis = np.linalg.norm(inclined_rect[i] - quad[0]) + \
              np.linalg.norm(inclined_rect[(i + 1) % 4] - quad[1]) + \
              np.linalg.norm(inclined_rect[(i + 2) % 4] - quad[2]) + \
              np.linalg.norm(inclined_rect[(i + 3) % 4] - quad[3])
        if dis < min_dis:
            first_ind = i
            min_dis = dis
    assert first_ind != -1
    return inclined_rect[(first_ind, (first_ind + 1) % 4,(first_ind + 2) % 4, (first_ind + 3) % 4 ), :]

def quads2inclinedRects(quads):
    """convert quad to inclined rect
    Return:
        inclined_rects (ndarray): N * 5 , (ind, left-upper.x, left-upper.y, right-upper.x, right-upper.y,
        dis(right-upper, right-down)
        ))
    """
    inclined_rects = np.zeros((len(quads), 5), dtype=np.float32)
    for ind, quad in enumerate(quads):
        rect = cv2.minAreaRect(quad.reshape((4, 2)))
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        box = find_order_boxes(box, quad)
        assert polygon_area(box) < 0
        inclined_rects[ind, :2] = box[0, :]
        inclined_rects[ind, 2:4] = box[1, :]
        inclined_rects[ind, 4] = np.linalg.norm(box[1] - box[2])
    return inclined_rects

def inclined_rect_transform(gt_boxes, gt_quads):
    """1. use ex_rois to scale gt_quads
       2. use gt_boxes(gt_rois) to scale gt_quads
       now we use 2"""
    # gt_inclined_rects = quads2inclinedRects(gt_quads)

    # gt_quads = gt_quads.reshape((-1, 4, 2))
    # gt_inclined_rects = generate_rbox(gt_quads)

    gt_boxes_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_boxes_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_boxes_ctr_x = gt_boxes[:, 0] + 0.5 * gt_boxes_widths
    gt_boxes_ctr_y = gt_boxes[:, 1] + 0.5 * gt_boxes_heights

    # gt_x1 = gt_inclined_rects[:, 0, 0]
    # gt_y1 = gt_inclined_rects[:, 0, 1]
    # gt_x2 = gt_inclined_rects[:, 1, 0]
    # gt_y2 = gt_inclined_rects[:, 1, 1]
    gt_x1 = gt_quads[:, 0]
    gt_y1 = gt_quads[:, 1]
    gt_x2 = gt_quads[:, 2]
    gt_y2 = gt_quads[:, 3]
    # gt_heights = gt_inclined_rects[:, 4]
    # gt_heights = np.linalg.norm(gt_inclined_rects[:, 2] - gt_inclined_rects[:, 1], axis=1)
    gt_heights = np.linalg.norm(gt_quads[:, 2:4] - gt_quads[:, 4:6], axis=1)

    targets_dx1 = (gt_x1 - gt_boxes_ctr_x) / gt_boxes_widths
    targets_dy1 = (gt_y1 - gt_boxes_ctr_y) / gt_boxes_heights
    targets_dx2 = (gt_x2 - gt_boxes_ctr_x) / gt_boxes_widths
    targets_dy2 = (gt_y2 - gt_boxes_ctr_y) / gt_boxes_heights
    targets_dh = np.log( gt_heights / gt_boxes_heights)

    targets = np.vstack(
        (targets_dx1, targets_dy1, targets_dx2, targets_dy2, targets_dh)).transpose()
    return targets

def inclined_rect_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
         return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    # anchor is first and second point in roi

    dx1 = deltas[:, 0::5]
    dy1 = deltas[:, 1::5]
    dx2 = deltas[:, 2::5]
    dy2 = deltas[:, 3::5]
    dh = deltas[:, 4::5]

    pre_x1 = dx1 * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pre_y1 = dy1 * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pre_x2 = dx2 * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pre_y2 = dy2 * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pre_h = np.exp(dh) * heights[:, np.newaxis]

    pre_inclined_rect = np.zeros((pre_x1.shape[0], pre_x1.shape[1] * 5), dtype=pre_x1.dtype)
    pre_inclined_rect[:, 0::5] = pre_x1
    pre_inclined_rect[:, 1::5] = pre_y1
    pre_inclined_rect[:, 2::5] = pre_x2
    pre_inclined_rect[:, 3::5] = pre_y2
    pre_inclined_rect[:, 4::5] = pre_h

    # pre_inclined_rect, inds = filter_inclined_rect(pre_inclined_rect)
    pre_inclined_rect = quad_transform_warp(pre_inclined_rect)

    # return pre_inclined_rect, inds
    return pre_inclined_rect

def quad_transform_warp(inclined_rect):
    """a warp function"""
    assert inclined_rect.shape[1] % 5 == 0
    quad = np.zeros((inclined_rect.shape[0], inclined_rect.shape[1] / 5 * 8), dtype=inclined_rect.dtype)
    print quad.shape
    for i in range(inclined_rect.shape[0]):
        rot_rect = quad_transform_inv(inclined_rect[i, 5:10])
        quad[i, 8:16] = rot_rect.reshape(1, -1)
    return quad

def quad_transform_inv(inclined_rect):
    """a test function to recovery inclined rectangle four vertexes crds from
    [x1, y1, x2, y2, height] used in RRCNN
    my method is very ugly"""
    axis_crds, theta = align_axis_crds(inclined_rect)
    if (axis_crds==0).all():
        return np.zeros((4, 2))
    if theta > 0: #anticlockwise
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    else: # clockwise
        rot_mat = np.array([[np.cos(-theta), np.sin(-theta)],
                            [-np.sin(-theta), np.cos(-theta)]], dtype=np.float32)
    deltas_xy = axis_crds.transpose() - np.array([[inclined_rect[0]], [inclined_rect[1]]])
    deltas_xy[1, :] *= -1
    rot_rect = np.dot(rot_mat, deltas_xy)
    rot_rect[1, :] *= -1
    rot_rect += np.array([[inclined_rect[0]], [inclined_rect[1]]])
    rot_rect = rot_rect.transpose()
    assert abs(rot_rect[0, 0] - inclined_rect[0]) < 1 and abs(rot_rect[0, 1] - inclined_rect[1]) < 1
    assert abs(rot_rect[1, 0] - inclined_rect[2]) < 1 and abs(rot_rect[1, 1] - inclined_rect[3]) < 1
    return rot_rect

def align_axis_crds(inclined_rect):
    width = math.sqrt( (inclined_rect[3] - inclined_rect[1])**2 + (inclined_rect[2] - inclined_rect[0])**2 )
    height = inclined_rect[4]
    axis_crds = np.zeros((4, 2), dtype= np.float32)

    # if inclined_rect[0] > inclined_rect[2]:
    #     return axis_crds, 0
    #(x1, y1)
    axis_crds[0, :] = inclined_rect[:2]
    #(x2, y2)
    axis_crds[1, 0] = inclined_rect[0] + width
    axis_crds[1, 1] = inclined_rect[1]
    #(x3, y3)
    axis_crds[2, 0] = axis_crds[1, 0]
    axis_crds[2, 1] = axis_crds[1, 1] + height
    #(x4, y4)
    axis_crds[3, 0] = axis_crds[0, 0]
    axis_crds[3, 1] = axis_crds[0, 1] + height

    theta = np.arcsin((inclined_rect[1] - inclined_rect[3] )/ width)

    return axis_crds, theta

def clip_quads(quads, im_shape):
    """
    Clip quads to image boundaries.
    """

    # x1 >= 0
    quads[:, 0::8] = np.maximum(np.minimum(quads[:, 0::8], im_shape[1] - 1), 0)
    # y1 >= 0
    quads[:, 1::8] = np.maximum(np.minimum(quads[:, 1::8], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    quads[:, 2::8] = np.maximum(np.minimum(quads[:, 2::8], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    quads[:, 3::8] = np.maximum(np.minimum(quads[:, 3::8], im_shape[0] - 1), 0)
    # x3 < im_shape[1]
    quads[:, 4::8] = np.maximum(np.minimum(quads[:, 4::8], im_shape[1] - 1), 0)
    # y3 < im_shape[0]
    quads[:, 5::8] = np.maximum(np.minimum(quads[:, 5::8], im_shape[0] - 1), 0)
    # x4 > 0
    quads[:, 6::8] = np.maximum(np.minimum(quads[:, 6::8], im_shape[1] - 1), 0)
    # y4 < im_shape[0]
    quads[:, 7::8] = np.maximum(np.minimum(quads[:, 7::8], im_shape[0] - 1), 0)

    return quads


def filter_inclined_rect(pre_inclined_rect):
    """some pre_inclined_rect is not regular, such x1 > x2"""
    inds = np.where(pre_inclined_rect[:, 7] > pre_inclined_rect[:, 5])
    return pre_inclined_rect[inds], inds