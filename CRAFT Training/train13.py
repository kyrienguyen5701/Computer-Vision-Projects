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
from data_loader import Synth80k, ICDAR2013
from loss import Loss
from torchvision.transforms import transforms
from craft import CRAFT
from torch.autograd import Variable
from torch.optim import lr_scheduler
from test import copyStateDict, test

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'

    print('Load the synthetic data ...')
    data_loader = Synth80k('D:/Datasets/SynthText')
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )
    batch_syn = iter(train_loader)

    print('Prepare the net ...')
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load('./weigths/synweights/0.pth')))
    net.to(device)
    data_parallel = False
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        data_parallel = True
    cudnn.benchmark = False

    print('Load the real data')
    real_data = ICDAR2013(net, 'D:/Datasets/ICDAR_2013')
    real_data_loader = torch.utils.data.DataLoader(
        real_data,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )

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
        for i, (real_images, real_region_scores, real_affinity_scores, real_confidence_mask) in enumerate(real_data_loader):
            syn_images, syn_region_scores, syn_affinity_scores, syn_confidence_mask = next(batch_syn)
            images = Variable(torch.cat((syn_images, real_images), 0).type(torch.FloatTensor))
            region_scores = Variable(torch.cat((syn_region_scores, real_region_scores), 0).type(torch.FloatTensor))
            affinity_scores = Variable(torch.cat((syn_affinity_scores, real_affinity_scores), 0).type(torch.FloatTensor))
            confidence_mask = Variable(torch.cat((syn_confidence_mask, real_confidence_mask), 0).type(torch.FloatTensor))
            if use_cuda:
                images, region_scores, affinity_scores, confidence_mask = images.cuda(), region_scores.cuda(), affinity_scores.cuda(), confidence_mask.cuda()

            pred_scores, _ = net(images)
            pred_region_scores = pred_scores[:, :, :, 0]
            pred_affinity_scores = pred_scores[:, :, :, 1]
            if use_cuda:
                pred_region_scores, pred_affinity_scores = pred_region_scores.cuda(), pred_affinity_scores.cuda()
            loss = criterion(region_scores, affinity_scores, pred_region_scores, pred_affinity_scores, confidence_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value += loss.item()

            if (i + 1) % 10 == 0:
                pause = time.time()
                print('Epoch {}:({}/{}) batch\nTraining time for 10 batches: {}\nTraining loss: {}'
                    .format(epoch, i + 1, len(train_loader), pause - start, loss_value/10))
                loss_value = 0

            if i + 1 == len(real_data_loader):
                print('Saving state dict, version', epoch)
                torch.save(net.state_dict(), './models/realweights/Epoch_{}.pth'.format(epoch + 1))
