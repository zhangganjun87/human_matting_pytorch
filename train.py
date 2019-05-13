#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhangganjun
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import dataset
import network

def get_args():
    parser = argparse.ArgumentParser(description='Fast matting')
    parser.add_argument('--dataDir', default='./data/',help='data root dir')
    parser.add_argument('--saveDir', default='./ckpt', help='model dir')
    parser.add_argument('--trainData', default='human_matting_data', help='train dataset name')
    parser.add_argument('--load', default= 'human_matting', help='save model')

    parser.add_argument('--finetuning', action='store_true', default=False, help='finetuning the training')
    parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')

    parser.add_argument('--nThreads', type=int, default=4, help='number of threads for data loading')
    parser.add_argument('--train_batch', type=int, default=8, help='input batch size for train')
    parser.add_argument('--patch_size', type=int, default=256, help='patch size for train')


    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lrDecay', type=int, default=100)
    parser.add_argument('--lrdecayType', default='keep')
    parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--save_epoch', type=int, default=1, help='number of epochs to save model')

#    parser.add_argument('--train_phase', default= 'end_to_end', help='train phase')
    
    args = parser.parse_args()
    print(args)
    return args

def set_lr(args, epoch, optimizer):
    
    lrDecay = args.lrDecay
    decayType = args.lrdecayType
    if decayType == 'keep':
        lr = args.lr
    elif decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2**epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'poly':
        lr = args.lr * math.pow((1 - epoch / args.nEpochs), 0.9)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr 

class Train_Log():
    def __init__(self, args):
        self.args = args
        
        self.save_dir = os.path.join(args.saveDir, args.load)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)
            
        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')
            
    def save_model(self, model, epoch):
        lastest_out_path = '{}/ckpt_lastest.pth'.format(self.save_dir_model)
        torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),}, lastest_out_path)
    
        model_out_path = '{}/model_obj.pth'.format(self.save_dir_model)
        torch.save(model, model_out_path)
        
    def load_model(self, model):
        lastest_out_path = '{}/ckpt_lastest.pth'.format(self.save_dir_model)
        ckpt = torch.load(lastest_out_path)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        print('=> loaded checkpoint "{}"(epoch{})'.format(lastest_out_path, ckpt['epoch']))
        
        return start_epoch, model
    
def loss_function(args, img, trimap_pre, alpha_pre, alpha_gt):
    # -------------------------------------
    # classification loss L_t
    # ------------------------
    # Cross Entropy 
    # criterion = nn.BCELoss()
    # trimap_pre = trimap_pre.contiguous().view(-1)
    # trimap_gt = trimap_gt.view(-1)
    # L_t = criterion(trimap_pre, trimap_gt)

#        criterion = nn.CrossEntropyLoss()
#        L_t = criterion(trimap_pre, trimap_gt[:,0,:,:].long())
    
    
    # prediction loss L_p
    eps = 1e-6
    #l_alpha
    L_alpha = torch.sqrt(torch.pow(alpha_pre - alpha_gt, 2.) + eps).mean()
    
    #L_composition
    fg = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img
    fg_pre = torch.cat((alpha_pre, alpha_pre, alpha_pre), 1) * img
    L_composition = torch.sqrt(torch.pow(fg - fg_pre, 2.) + eps).mean()
    
    L_p = 0.5 * L_alpha + 0.5 * L_composition
    
    #train phase
    loss = L_p
    
    return loss, L_alpha, L_composition
    
    
def main():
    
    print('=========== loading args')
    args = get_args()
    
    print('=========== Enviroment init')
    if args.without_gpu:
        print('use cpu')
        device = torch.device('cpu')
        
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print('no gpu is available')
            
    print('============ Building model')
    model = network.net()
    model.to(device)
    
    print('============ Loading datasets...')
    train_data = getattr(dataset, args.trainData)(root_dir=args.dataDir, patch_size=args.patch_size)
    
    trainloader = DataLoader(train_data, 
                             batch_size=args.train_batch,
                             drop_last=True,
                             shuffle=True,
                             num_workers=args.nThreads,
                             pin_memory=True)
    print('============ Set optimizer  ...')
    lr = args.lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr = lr, betas=(0.9, 0.999), weight_decay=0.0005)
    print('============= start train....')
    start_epoch = 1
    trainlog = Train_Log(args)
    if args.finetuning:
        start_epoch, model = trainlog.load_model(model)
        
    model.train()
    for epoch in range(start_epoch, args.nEpochs+1):
        loss_ = 0
        L_alpha_ = 0
        L_composition_ = 0
        L_cross_ = 0
        if args.lrdecayType != 'keep':
            lr = set_lr(args, epoch, optimizer)
            
        t0 = time.time()
        for i, sample_batched in enumerate(trainloader):
            img, alpha_gt = sample_batched['image'], sample_batched['alpha']
            img, alpha_gt = img.to(device), alpha_gt.to(device)
            if i%1000 == 0:
                print('processed ' + str(i) + 'th batch....')
            trimap_pre, alpha_pre = model(img)
            loss, L_alpha, L_composition = loss_function(args, img, trimap_pre,
                                                         alpha_pre, alpha_gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_ += loss.item()
            L_alpha_ += L_alpha.item()
            L_composition_ += L_composition.item()
            
        t1 = time.time()
        
        if epoch % args.save_epoch ==0:
            
            speed = (t1 - t0) / args.save_epoch
            
            loss_ = loss_ / (i+1)
            L_alpha_ = L_alpha_ / (i+1)
            L_composition_ = L_composition_ / (i+1)
            
            log = '[{}/{}] \tLr:{:.5f}\nloss:{:.5f}\tloss_p:{:.5f}\tspeed:{}'.format(
                    epoch, args.nEpochs, lr, loss_, L_alpha_+L_composition_,speed)
            
            print(log)
#            trainlog.save_log(log)
            trainlog.save_model(model, epoch)
            

if __name__ == '__main__':
    main()
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    






























