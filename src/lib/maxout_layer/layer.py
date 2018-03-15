
import caffe
import numpy as np

class MaxoutLayer(caffe.Layer):
    """multi-scales test Nocs:
    after ROIPooling , got a (num_scales * num_rois, c , Pool_h, Poow) size
    we need a maxout operator according Nocs"""
    def setup(self, bottom, top):
        if len(bottom[0].data.shape) == 2:
            bottom[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1], 1, 1)
        rois_pool = bottom[0].data
        im_scales = bottom[1].data[0, 2:]
        assert  rois_pool.shape[0] % len(im_scales) == 0

        rois_per_scale = rois_pool.shape[0] / len(im_scales)

        top[0].reshape(rois_per_scale, rois_pool.shape[1], rois_pool.shape[2], rois_pool.shape[3])

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        if len(bottom[0].data.shape) == 2:
            bottom[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1], 1, 1)
        rois_pool = bottom[0].data
        im_scales = bottom[1].data[0, 2:]
        assert  rois_pool.shape[0] % len(im_scales) == 0

        rois_per_scale = rois_pool.shape[0] / len(im_scales)
        max_pool = np.zeros((rois_per_scale, rois_pool.shape[1], rois_pool.shape[2], rois_pool.shape[3]), dtype=rois_pool.dtype)
        # max_pool = -1
        for i in xrange(len(im_scales)):
            max_pool = np.maximum(max_pool, rois_pool[rois_per_scale * i: rois_per_scale * (i + 1), :, :, :])
        top[0].reshape(*max_pool.shape)
        top[0].data[...] = max_pool

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
