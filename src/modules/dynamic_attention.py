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


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging import INFO, WARN


class DynamicAttention(nn.Module):
    def __init__(self, d_model,
                 d_per_capsule,
                 n_capsule_per_category=2,
                 n_category=3,
                 n_iteration=3):
        super().__init__()

        self.d_model = d_model
        self.d_per_capsule = d_per_capsule
        self.n_capsule_per_category = n_capsule_per_category
        self.n_category = n_category
        self.n_iteration = n_iteration

        self.linear_key = nn.Linear(d_model, d_per_capsule)
        self.linear_value = nn.Linear(d_model, d_per_capsule)
        self.linear_query = nn.Linear(d_model, d_per_capsule)

    @staticmethod
    def squash(tensor, dim=-1, eps=1e-8):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        norm = torch.sqrt(squared_norm) + eps
        return scale * tensor / norm

    def forward(self, query, key, value, mask=None, cache=None):
        """
        Args:
        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, d_model]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`

        Returns: Tensor. [batch_size, length, num_out_caps, dim_out_caps]

        """
        batch_size, num_in_caps, dim_in_caps = inputs_u.size()
        length = context_sequence.size(1)
        # Compute u_hat: [batch_size, num_in_caps, num_out_caps, dim_out_caps]
        if cache is not None:
            priors_u_hat = cache
        else:
            priors_u_hat = self.compute_caches(inputs_u)

        # Initialize logits
        # logits_b: [batch_size, length, num_in_caps, num_out_caps]
        logits_b = inputs_u.new_zeros(batch_size, length, num_in_caps, self.num_out_caps)
        # [batch, 1, num_in_caps, 1]
        routing_mask = inputs_mask[:, None, :, None].expand_as(logits_b)

        # Routing
        for i in range(self.num_iterations):
            # probs: [batch_size, length, num_in_caps, num_out_caps]
            if inputs_mask is not None:
                logits_b = logits_b.masked_fill(routing_mask, -1e18)
            probs_c = F.softmax(logits_b, dim=-1)

            # # [batch, num_out_caps, length,
            # _interm = probs_c.permute([0, 3, 1, 2]) @ prior_u_hat.transpose(1, 2))
            # outputs_v: [batch_size, length, num_out_caps, dim_out_caps]
            outputs_v = self.squash((probs_c.unsqueeze(-1) * priors_u_hat.unsqueeze(1)).sum(2))

            if i != self.num_iterations - 1:
                # delta_logits: [batch_size, length, num_in_caps, num_out_caps]
                delta_logits = self.compute_delta_sequence(
                    priors_u_hat, outputs_v, context_sequence
                )
                logits_b = logits_b + delta_logits

        # outputs_v: [batch_size, length, num_out_caps, dim_out_caps]
        return outputs_v, probs_c

    def compute_caches(self, inputs_u):
        batch_size, num_in_caps, dim_in_caps = inputs_u.size()
        if self.share_route_weights_for_in_caps:
            inputs_u_r = inputs_u.view(batch_size * num_in_caps, dim_in_caps)
            route_weight_r = self.route_weights.transpose(0, 1).reshape(dim_in_caps, -1)
            priors_u_hat = inputs_u_r @ route_weight_r
            priors_u_hat = priors_u_hat.view(batch_size, num_in_caps, self.num_out_caps, -1)
        else:
            priors_u_hat = (inputs_u[:, :, None, None, :] @ self.route_weights[None, :, :, :,
                                                            :]).squeeze(-2)
        return priors_u_hat