from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.vocabulary import PAD
from src.utils import GlobalNames


class Criterion(nn.Module):
    """ Class for managing loss computation.

    """

    def _compute_loss(self, inputs, labels, **kwargs):
        """
        Compute the loss. Subclass must override this method.

        Args:
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        Returns:
            A non-reduced FloatTensor with shape (batch, )
        """
        raise NotImplementedError

    def forward(self, inputs, labels, normalization=1.0, reduce=True, **kwargs):
        """
        Compute loss given inputs and labels.

        Args:
            inputs: Input tensor of the criterion.
            labels: Label tensor of the criterion.
            reduce: Boolean value indicate whether the criterion should reduce the loss along the batch. If false,
                the criterion return a FloatTensor with shape (batch, ), otherwise a scalar.
            normalization: Normalization factor of the loss. Should be a float scalar or a FloatTensor with shape
                (batch, )
        """
        loss = self._compute_loss(inputs, labels, **kwargs).div(normalization)  # (batch, )

        if reduce:
            loss = loss.sum()

        return loss


class NMTCriterion(Criterion):
    """ A common used criterion for neural machine translation

    NMTCriterion is used for MLE training given golden target sample. Additional label_smoothing
    is supported.
    """

    def __init__(self, padding_idx=PAD, label_smoothing=0.0):

        super().__init__()

        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing

        if label_smoothing > 0:

            self.criterion = nn.KLDivLoss(size_average=False, reduce=False)

        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=padding_idx, reduce=False)

        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):

        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.

        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 2))
        one_hot[0][self.padding_idx] = 0

        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _compute_loss(self, inputs, labels, **kwargs):

        """
        Args:
            inputs (..., K): Expect logarithm probabilities.
            labels (...,): Index tensor. Should be the same size as inputs except the last dimension.
        """

        batch_size = labels.size(0)

        scores = self._bottle(inputs)  # [batch_size * seq_len, d_words]

        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)

        if self.confidence < 1:
            # N: the number of samples
            # M: the number of labels
            tdata = gtruth.detach()

            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()  # mask of PAD

            one_hot = self._smooth_label(num_tokens)  # Do label smoothing
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if mask.numel() > 0:
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_.detach()

        loss = self.criterion(scores, gtruth).view((batch_size, -1)).sum(-1)

        return loss


class MultiCriterion(nn.Module):
    """
        Class for easily managing multiple criterions, which receives multiple instances of `
    Criterion`, computing loss respectively and summing them.
    """

    def __init__(self, weights=None, **named_criterions):
        """
        Args:
            weights (dict: weights for each criterions
            **named_criterions: {criterion_name: criterion}
        Notes:
            Each criterion must implement `_compute_loss(**kwargs)`
        """
        super(MultiCriterion, self).__init__()

        # remove None items.
        for kk in list(named_criterions.keys()):
            if named_criterions[kk] is None:
                named_criterions.pop(kk)

        for name, criterion in named_criterions.items():
            assert hasattr(criterion, "_compute_loss"), \
                "{} ({}) must have method \"_compute_loss\"".format(criterion, name)

        self.criterions = nn.Sequential(OrderedDict(named_criterions))
        self.num_criterions = len(self.criterions)
        self.names = list(named_criterions.keys())

        self.weights = weights if weights is not None \
            else {name: 1. for name in self.names}

    def add(self, name, criterion, weight=None):
        self._assert(name, criterion)

        self.names.append(name)
        self.criterions.add_module(name, criterion)
        if weight is None:
            weight = 1.
        self.weights[name] = weight

    @staticmethod
    def _assert(name, criterion):
        assert hasattr(criterion, "_compute_loss"), \
            "{} ({}) must have method \"_compute_loss\"".format(criterion, name)

    def compute_loss(self, **named_states):
        """
        Compute each loss respectively, and summing them.

        Args:
            named_states (dict): key-value params corresponds to specific criterion (name)
        Returns:
            losses (dict): dictionary of each named losses and a final loss summing of them.
        """
        losses = dict()
        list_losses = []
        for name, criterion in self.criterions._modules.items():
            # scalar
            if named_states[name] is None:
                loss = torch.tensor(0.0)
                if GlobalNames.USE_GPU:
                    loss = loss.cuda()
            else:
                loss = criterion._compute_loss(**named_states[name])  # [batch,]
            losses[name] = loss
            if named_states[name] is not None and named_states[name]['update'] is True:
                list_losses.append(self.weights[name] * loss)

        # losses["loss"] = torch.cat(list_losses, dim=-1).sum(-1)  # scalar
        losses["loss"] = sum(list_losses)  # [batch, ]

        return losses

    def forward(self, normalization=1.0, reduce=True, **named_states):
        """
        Args:
            **named_states: inputs for criterions. Must match corresponding names.

        Returns:

        """
        losses = self.compute_loss(**named_states)  # dict of [batch, ]
        for kk in losses:
            losses[kk] = losses[kk].div(normalization)
            if reduce:
                losses[kk] = losses[kk].sum()
        return losses


class MSELoss(Criterion):
    def __init__(self, size_average=False):
        super().__init__()
        self.size_average = size_average

    def _compute_loss(self, inputs, labels, mask, **kwargs):
        """
        Args:
            inputs: binary logits, [batch, length, dim]
            labels: [batch, length, dim]
            mask: [batch, length]
            **kwargs:

        Returns:

        """
        batch_size = inputs.size(0)
        mask = mask.float()
        # [batch, length, length]
        loss = F.mse_loss(inputs, labels, size_average=self.size_average, reduce=False).mean(-1)
        # [batch, ]
        loss = (loss * mask).view(batch_size, -1).sum(-1) / mask.view(batch_size, -1).sum(-1)
        return loss
