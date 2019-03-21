# -*- coding: utf-8 -*-
# @Time    : 2019/1/11 20:58
# @Author  : liangxhao
# @Email   : liangxhao@gmail.com
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax


def crf_processing(image, label, soft_label=False):
    crf = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
    if not soft_label:
        unary = unary_from_labels(label, 2, gt_prob=0.9, zero_unsure=False)
    else:
        if len(label.shape) == 2:
            p_neg = 1.0 - label
            label = np.concatenate((p_neg[..., np.newaxis], label[..., np.newaxis]), axis=2)
        label = label.transpose((2, 0, 1))
        unary = unary_from_softmax(label)
    crf.setUnaryEnergy(unary)
    crf.addPairwiseGaussian(sxy=(3, 3), compat=3)
    crf.addPairwiseBilateral(sxy=(40, 40), srgb=(5, 5, 5), rgbim=image, compat=10)
    crf_out = crf.inference(5)

    # Find out the most probable class for each pixel.
    return np.argmax(crf_out, axis=0).reshape((image.shape[0], image.shape[1]))