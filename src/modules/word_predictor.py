# -*- coding: UTF-8 -*- 

# Copyright 2018, Natural Language Processing Group, Nanjing University, 
#
#       Author: Zheng Zaixiang
#       Contact: zhengzx@nlp.nju.edu.cn 
#           or zhengzx.142857@gmail.com
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.vocabulary import PAD
from src.models import dl4mt
from src.modules.criterions import NMTCriterion
from src.utils import GlobalNames


class WordPredictor(nn.Module):
    def __init__(self, input_size, generator=None, **config):
        super().__init__()
        if generator is not None:
            self.generator = generator
        else:
            self.generator = dl4mt.Generator(n_words=config['n_tgt_vocab'],
                                             hidden_size=config['d_word_vec'],
                                             padding_idx=PAD)
        self.linear = nn.Linear(input_size, config['d_word_vec'])

    def forward(self, hiddens, logprob=True):
        logits = F.tanh(self.linear(hiddens))
        return self.generator(logits, logprob)


def generate_wpe_labels(labels):
    """
    :param probs: [batch, seq_len, n_words]
    :param labels: [batch, seq_len]
    :return: out: labels with ignored position being padding_idx. [batch, 1, seq_len]
    """
    # [batch, 1, seq_len]
    out = labels.unsqueeze(1)

    return out


def get_average_score(mask):
    mask = mask.float()
    scores = mask / mask.sum(-1, keepdim=True)
    scores = torch.where(torch.isnan(scores),
                         torch.zeros_like(scores),
                         scores)
    return scores


def convert_to_past_labels(labels, padding_idx=PAD):
    """
    Args:
        padding_idx:
        labels: [batch, seq_len]

    Returns:
        descending labels .
            [batch, seq_len, seq_len]
    """
    batch_size, seq_len = labels.size()
    seq_mask = labels.ne(padding_idx)

    # use upper triangle to masking in descending manner
    # [batch, seq_len, seq_len]
    step_mask = torch.tril(labels.new_ones(seq_len, seq_len), 0).byte()
    mask = step_mask.unsqueeze(0) * seq_mask.unsqueeze(1)
    scores = get_average_score(mask)

    # tile through timesteps
    # [batch, seq_len, seq_len]
    past_labels = labels.unsqueeze(1).repeat(1, seq_len, 1)
    # masking padded position by padding_idx
    past_labels.masked_fill_(1 - mask, padding_idx)

    return past_labels, scores


def convert_to_future_labels(labels, padding_idx=PAD):
    """
    Args:
        padding_idx:
        labels: [batch, seq_len]

    Returns:
        future labels .
            [batch, seq_len, seq_len]
    """
    batch_size, seq_len = labels.size()
    seq_mask = labels.ne(padding_idx)

    # use upper triangle to masking in descending manner
    # [batch, seq_len, seq_len]
    step_mask = torch.triu(labels.new_ones(seq_len, seq_len), 1).byte()
    mask = step_mask.unsqueeze(0) * seq_mask.unsqueeze(1)
    scores = get_average_score(mask)

    # tile through timesteps
    # [batch, seq_len, seq_len]
    future_labels = labels.unsqueeze(1).repeat(1, seq_len, 1)
    # masking padded position by padding_idx
    future_labels.masked_fill_(1 - mask, padding_idx)

    return future_labels, scores


class MultiTargetNMTCriterion(NMTCriterion):
    def _construct_target(self, targets, target_scores, num_tokens):
        """
        Args:
            targets: A Tensor with shape [batch*length, max_target] represents the indices of
                targets in the vocabulary.
            target_scores: A Tensor with shape [batch*length, max_target] represents the
                probabilities of targets in the vocabulary.
            num_tokens: An Integer represents the total number of words.

        Returns:
            A Tensor with shape [batch*length, num_tokens].
        """
        # Initialize a temporary tensor.
        if self.confidence < 1:
            tmp = self._smooth_label(num_tokens)  # Do label smoothing
            target_scores = target_scores * self.confidence
        else:
            tmp = torch.zeros(1, num_tokens)
        if targets.is_cuda:
            tmp = tmp.cuda()

        pad_positions = torch.nonzero(target_scores.sum(-1).eq(0)).squeeze()

        # [batch*length, num_tokens]
        tmp = tmp.repeat(targets.size(0), 1)

        if torch.numel(pad_positions) > 0:
            tmp.index_fill_(0, pad_positions, 0.)
        tmp.scatter_(1, targets, 0.)
        tmp.scatter_add_(1, targets, target_scores)

        return tmp

    def _compute_loss(self, inputs, labels, **kwargs):
        """
        Args:
            inputs: [batch, length, num_tokens]
            labels: [batch, length, max_target]
            **kwargs:

        Returns:
            A Tensor with shape of [batch,].
        """
        batch_size = labels.size(0)
        scores = self._bottle(inputs)  # [batch_size * seq_len, num_tokens]
        num_tokens = scores.size(-1)

        targets = self._bottle(labels)  # [batch_size * seq_len, max_target]
        target_scores = self._bottle(kwargs['target_scores'])

        # [batch*length, num_tokens]
        gtruth = self._construct_target(targets, target_scores, num_tokens)

        # [batch,]
        loss = self.criterion(scores, gtruth)
        loss = loss.sum(-1).view(batch_size, -1)  # [batch, seq_len]
        length_norm = kwargs['target_scores'].sum(-1).ne(0).float().sum(-1)  # [batch, ]
        loss = loss.sum(-1).div(length_norm)

        return loss

