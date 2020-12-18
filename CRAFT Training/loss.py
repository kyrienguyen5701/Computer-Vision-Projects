import numpy as np
import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, use_gpu = True):
        super(Loss,self).__init__()

    # filter out some negative loss
    def filter_loss(self, pred_loss, loss_label):
        batch_size = pred_loss.shape[0]
        sum_loss = torch.mean(pred_loss.view(-1))*0
        pred_loss = pred_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        for i in range(batch_size):
            average_number = 0
            loss = torch.mean(pred_loss.view(-1)) * 0
            positive_pixels = pred_loss[i][(loss_label[i] >= 0.1)]
            negative_pixels = pred_loss[i][(loss_label[i] < 0.1)]
            average_number += len(positive_pixels)
            if len(positive_pixels) != 0:
                positive_loss = torch.mean(positive_pixels)
                sum_loss += positive_loss
                if len(negative_pixels) < 3 * len(positive_pixels):
                    negative_loss = torch.mean(negative_pixels)
                    average_number += len(negative_pixels)
                else:
                    negative_loss = torch.mean(torch.topk(negative_pixels, 3 * len(positive_pixels))[0])
                    average_number += 3 * len(positive_pixels)
                sum_loss += negative_loss
            else:
                negative_loss = torch.mean(torch.topk(pred_loss[i], 500)[0])
                average_number += 500
                sum_loss += negative_loss

        return sum_loss

    def forward(self, gt_region_scores, gt_affinity_scores, pred_region_scores, pred_affinity_scores, confidence_mask):
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        assert pred_region_scores.size() == gt_region_scores.size() and pred_affinity_scores.size() == gt_affinity_scores.size()
        loss1 = loss_fn(pred_region_scores, gt_region_scores)
        loss2 = loss_fn(pred_affinity_scores, gt_affinity_scores)
        loss_r = torch.mul(loss1, confidence_mask)
        loss_a = torch.mul(loss2, confidence_mask)
        r_loss = self.filter_loss(loss_r, gt_region_scores)
        a_loss = self.filter_loss(loss_a, gt_affinity_scores)
        return r_loss/loss_r.shape[0] + a_loss/loss_a.shape[0]