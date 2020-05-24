# -*- coding: UTF-8 -*- 
# 
# MIT License
#
# Copyright (c) 2018 the xnmt authors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#       author: Zaixiang Zheng
#       contact: zhengzx@nlp.nju.edu.cn 
#           or zhengzx.142857@gmail.com
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def get_mask(seq, mode="upper"):
    ''' Get an attention mask for applying directional self-attention.

    :param seq: Input sequence.
        with shape [batch_size, time_steps, dim]
    '''
    assert seq.dim() == 3
    shape = (seq.size(0), seq.size(1), seq.size(1))
    if mode == "upper":
        mask = np.triu(np.ones(shape), k=1).astype('uint8')
    elif mode == "lower":
        mask = np.tril(np.ones(shape), k=-1).astype('uint8')
    elif mode == "diag":
        mask = np.diag(np.ones(shape)).astype('uint8')
    mask = torch.from_numpy(mask)
    if seq.is_cuda:
        mask = mask.cuda()
    return mask


def get_average(mask):
    mask = mask.float()
    scores = mask / mask.sum(-1, keepdim=True)
    scores = torch.where(torch.isnan(scores),
                         torch.zeros_like(scores),
                         scores)
    return scores


def get_prev_sequence_average(tensor):
    b, l, d = tensor.size()

    mask = tensor.new_tensor(torch.tril(torch.ones([l, l]), -1))
    mask = get_average(mask)
    mask = mask[None, :, :].repeat(b, 1, 1)

    # [b, l, l] x [b, l, d]
    out = mask @ tensor
    return out


def get_sub_sequence_average(tensor):
    b, l, d = tensor.size()

    mask = tensor.new_tensor(torch.triu(torch.ones([l, l]), 0))
    mask = get_average(mask)
    mask = mask[None, :, :].repeat(b, 1, 1)

    # [b, l, l] x [b, l, d]
    out = mask @ tensor
    return out
