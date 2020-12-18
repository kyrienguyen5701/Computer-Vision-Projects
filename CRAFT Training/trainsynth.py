import os
import sys
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import cv2
import numpy as np
import time
from data_loader import Synth80k
from loss import Loss
from torchvision.transforms import transforms
from craft import CRAFT
from torch.autograd import Variable
from torch.optim import lr_scheduler

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'
    print('Load the data ...')

    data_loader = Synth80k('D:/SynthText')
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )
    print('Prepare the net ...')
    net = CRAFT().to(device)
    data_parallel = False
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        data_parallel = True
    cudnn.benchmark = False

    lr = .0001
    epochs = 15
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//3], gamma=0.1)
    criterion = Loss()

    print('Begin training ...')
    net.train()
    for epoch in range(epochs):
        scheduler.step()
        loss_value = 0
        start = time.time()
        for i, (images, gt_region_scores, gt_affinity_scores, confidence_mask) in enumerate(train_loader):
            images = Variable(images.type(torch.FloatTensor))
            gt_region_scores = Variable(gt_region_scores.type(torch.FloatTensor))
            gt_affinity_scores = Variable(gt_affinity_scores.type(torch.FloatTensor))
            confidence_mask = Variable(confidence_mask.type(torch.FloatTensor))
            if use_cuda:
                images, gt_region_scores, gt_affinity_scores, confidence_mask = images.cuda(), gt_region_scores.cuda(), gt_affinity_scores.cuda(), confidence_mask.cuda()
            
            pred_scores, _ = net(images)
            pred_region_scores = pred_scores[:, :, :, 0]
            pred_affinity_scores = pred_scores[:, :, :, 1]
            if use_cuda:
                pred_region_scores, pred_affinity_scores = pred_region_scores.cuda(), pred_affinity_scores.cuda()
            loss = criterion(gt_region_scores, gt_affinity_scores, pred_region_scores, pred_affinity_scores, confidence_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value += loss.item()

            if (i + 1) % 10 == 0:
                pause = time.time()
                print('Epoch {}:({}/{}) batch\nTraining time for 10 batches: {}\nTraining loss: {}'
                    .format(epoch, i + 1, len(train_loader), pause - start, loss_value/10))
                loss_value = 0

            if i + 1 == 500 and (epoch + 1) % 3 == 0:
                print('Saving state dict, version', (epoch + 1) // 3)
                torch.save(net.state_dict(), './weights/synweights/Epoch_{}.pth'.format(epoch + 1))