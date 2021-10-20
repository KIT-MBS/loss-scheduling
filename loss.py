#!/usr/bin/env python3
import math

import torch


# NOTE adapted from pytorch PR #1249 issuecomment 339904369 and CoinCheung/pytorch-loss
# NOTE like ce, expects unnormalized scores
def dice_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', smooth=1., eps=1e-5):
    input = torch.nn.functional.softmax(input, dim=1)

    if size_average is not None:
        raise
    if reduce is not None:
        raise

    assert input.dim() == 4

    encoded_target = torch.zeros_like(input)
    if ignore_index is not None:
        mask = target == ignore_index
        masked_target = target.clone().detach()
        masked_target[mask] = 0
        encoded_target.scatter_(1, masked_target.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand(input.size())
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
    assert input.size() == encoded_target.size()

    numerator = (input * encoded_target).sum(dim=(2, 3))
    denominator = (input*input + encoded_target*encoded_target)

    if ignore_index is not None:
        denominator[mask] = 0.

    denominator = denominator.sum(dim=3).sum(dim=2)

    if weight is not None:
        numerator = numerator * weight
        denominator = denominator * weight

    denominator = denominator.sum(1)
    numerator = numerator.sum(1)

    dice = 2 * ((numerator+smooth) / (denominator+smooth+eps))

    if reduction == 'mean':
        return 1. - dice.mean()
    else:
        return 1. - dice


class CompoundLoss(torch.nn.modules.loss._Loss):
    def __init__(self, loss1, loss2, weight=[1., 0.], kwargs1={}, kwargs2={}, size_average=None, reduce=None, reduction='mean', ignore_index=-100):
        super(CompoundLoss, self).__init__(size_average, reduce, reduction)
        kwargs1['ignore_index'] = ignore_index
        kwargs2['ignore_index'] = ignore_index
        self.loss1 = loss1(**kwargs1)
        self.loss2 = loss2(**kwargs2)
        self.w = weight
        return

    def reweight(self, weight):
        self.w = weight
        return

    def forward(self, input, target):
        return self.w[0] * self.loss1(input, target) + self.w[1] * self.loss2(input, target)


class GradientRescaledLoss(torch.nn.modules.loss._Loss):
    def __init__(self, loss, p=1.3, m_over_n=0.1, kwargs={}, size_average=None, reduce=None, reduction='mean', ignore_index=-100):
        super(MaxPooledLoss, self).__init__(size_average, reduce, reduction)
        self.loss = loss(ignore_index=ignore_index, reduction='none', **kwargs)
        self.p = p
        self.m_over_n = m_over_n
        return

    # NOTE catch all the boring cases instead of blowing up
    def forward(self, input, target):
        b, c, h, w = input.size()
        l = self.loss(input, target)
        assert l.size() == (b, h, w)
        l = l.flatten()
        n = h*w*b
        m = self.m_over_n * n
        assert self.p > 1
        assert math.floor(m) < n

        q = self.p/(self.p-1)
        gamma = n**(-1/q)
        tau = gamma * m**(-1/self.p)

        values, indices = torch.topk(l, k=math.floor(m)+1)
        alphastar = values[-1]

        weights = torch.where(l > alphastar, torch.full_like(l, tau), tau*(l/alphastar)**(q-1))
        weights = weights.detach()
        weights *= n/2

        if self.reduction == 'mean':
            return (weights*l).mean()
        return (weights*l).reshape(b, h, w)


class MaxPooledLoss(torch.nn.modules.loss._Loss):
    def __init__(self, loss, p=1.3, m_over_n=0.1, kwargs={}, size_average=None, reduce=None, reduction='mean', ignore_index=-100):
        super(MaxPooledLoss, self).__init__(size_average, reduce, reduction)
        self.loss = loss(ignore_index=ignore_index, reduction='none', **kwargs)
        self.p = p
        self.m_over_n = m_over_n
        return

    def forward(self, input, target):
        b, c, h, w = input.size()
        l = self.loss(input, target)
        assert l.size() == (b, h, w)
        l = l.flatten()
        n = h*w*b
        m = self.m_over_n * n
        assert self.p > 1
        assert math.floor(m) <= n

        q = self.p/(self.p-1)
        gamma = n**(-1/q)
        tau = gamma * m**(-1/self.p)

        # values, indices = torch.topk(l, n, largest=False)
        # values, indices = torch.topk(l, n)
        values, indices = torch.sort(l)
        # TODO there is a better way to do this
        i = 0
        c = m - n
        a = 0
        eta = None
        # for v in values:
        while True:
            v = values[i]
            i += 1
            c += 1
            a = a + v
            eta = c * v - a
            if eta > 0 or i == n:
                break
        if eta <= 0:
            i += 1
        alpha = (a/c)**(1/q)
        J = indices[i:n]

        weights = torch.zeros_like(l)
        weights[J] = tau
        if alpha > 0:
            notJ = indices[0:i]
            weights[notJ] = tau * (l[notJ]/alpha)**(q-1)

        if self.reduction == 'mean':
            return (weights * l).mean()
        return (weights * l).reshape(b, h, w)


class DiceLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(DiceLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return dice_loss(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)


class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction='none', ignore_index=ignore_index)
        return

    def forward(self, input, target):
        loss = self.ce(input, target)
        p = torch.exp(-loss)
        loss = loss * (1-p)**self.gamma

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


if __name__ == '__main__':
    pass
