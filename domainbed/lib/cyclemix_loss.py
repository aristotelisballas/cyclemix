from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
from collections import OrderedDict


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def find_same_indexes(y_task):
    y_tmp_np = y_task.cpu().detach().numpy()
    y_tmp = y_tmp_np.tolist()
    counts = {}
    same_indexes_tmp = {}
    dif_indexes = {}
    for i in y_tmp:
        counts[i] = y_tmp.count(i)
        same_indexes_tmp[i] = np.where(y_tmp_np == i)
        dif_indexes[i] = np.where(y_tmp_np != i)

    same_indexes_tmp = OrderedDict(sorted(same_indexes_tmp.items()))
    same_indexes = []
    for i in range(len(same_indexes_tmp.items())):
        if i in same_indexes_tmp.keys():
            same_indexes.append(torch.combinations(torch.tensor(same_indexes_tmp[i][0])))

    return same_indexes


def cyclemix_contra_loss(features: torch.Tensor,
            y_task: torch.Tensor,
            temperature: float,
            ):

    same_indexes = find_same_indexes(y_task)
    # batch_energy = layer_activations
    A = features

    A_n = torch.nn.functional.normalize(A)

    all_energy = torch.exp(torch.matmul(A_n, A_n.T))
    # all_energy = all_energy.fill_diagonal_(0)
    denominator = torch.sum(all_energy)

    loss = 0

    for i in range(len(same_indexes)):
        pos_energy = 0
        if len(same_indexes[i]) != 0:
            # print(i)
            for pair in same_indexes[i]:
                pos_energy += (torch.multiply(all_energy[(pair[0], pair[1])], 2) / temperature)
            class_loss = torch.log(torch.div(pos_energy, denominator))
            loss += class_loss

    return loss
