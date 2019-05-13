#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhangganjun
"""
import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, inSize, outSize, stride, expandRatio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        self.use_res_connect = self.stride == 1 and inSize == outSize
        
        self.conv = nn.Sequential(
                nn.Conv2d(inSize, inSize * expandRatio, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inSize * expandRatio),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(inSize * expandRatio, inSize * expandRatio, 3, stride, 1, groups=inSize * expandRatio, bias=False),
                nn.BatchNorm2d(inSize * expandRatio),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(inSize*expandRatio, outSize, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outSize),
                )
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class mobilenet_v2(nn.Module):
    def __init__(self, nInputChannels=3):
        super(mobilenet_v2, self).__init__()
        self.head_conv = nn.Sequential(
                nn.Conv2d(nInputChannels, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                )
        
        self.block_1 = InvertedResidual(32, 16, 1, 1)
        
        self.block_2 = nn.Sequential(
                InvertedResidual(16, 24, 2, 6),
                InvertedResidual(24, 24, 1, 6),
                )
        
        self.block_3 = nn.Sequential(
                InvertedResidual(24, 32, 2, 6),
                InvertedResidual(32, 32, 1, 6),
                InvertedResidual(32, 32, 1, 6),
                )
        
        self.block_4 = nn.Sequential(
                InvertedResidual(32, 64, 2, 6),
                InvertedResidual(64, 64, 1, 6),
                InvertedResidual(64, 64, 1, 6),
                InvertedResidual(64, 64, 1, 6),
                )
        
        self.block_5 = nn.Sequential( 
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)          
            )
        
        self.block_6 = nn.Sequential( 
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)          
            )
        
        self.block_7 = InvertedResidual(160, 320, 1, 6)
        
    def forward(self, x):
        x = self.head_conv(x)
        
        s1 = self.block_1(x)
        
        s2 = self.block_2(s1)
        
        s3 = self.block_3(s2)
        
        s4 = self.block_4(s3)
        s4 = self.block_5(s4)
        
        s5 = self.block_6(s4)
        s5 = self.block_7(s5)
        
        return s1, s2, s3, s4, s5
    

class T_mv2_unet(nn.Module):
    def __init__(self, classes=3):
        super(T_mv2_unet, self).__init__()
        
        #encoder
        self.feature = mobilenet_v2()
        
        #decoder
        self.s5_up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2,mode='bilinear'),
                nn.Conv2d(320, 96, 3, 1, 1),
                nn.BatchNorm2d(96),
                nn.ReLU()
                )
        self.s4_fusion = nn.Sequential(
                nn.Conv2d(96, 96, 3, 1, 1),
                nn.BatchNorm2d(96)
                )
        self.s4_up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(96, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU()
                )
        self.s3_fusion = nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.BatchNorm2d(32)
                )
        self.s3_up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2,mode='bilinear'),
                nn.Conv2d(32, 24, 3, 1, 1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                )
        self.s2_fusion = nn.Sequential(
                nn.Conv2d(24, 24, 3, 1, 1),
                nn.BatchNorm2d(24))
        self.s2_up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(24, 16, 3, 1, 1),
                nn.BatchNorm2d(16),
                nn.ReLU())
        self.s1_fusion = nn.Sequential(
                nn.Conv2d(16, 16, 3, 1, 1),
                nn.BatchNorm2d(16))
        self.last_conv = nn.Conv2d(16, classes, 3, 1, 1)
        self.last_up = nn.Upsample(scale_factor=2, mode='bilinear')
        
    def forward(self, input):
        
        s1, s2, s3, s4, s5 = self.feature(input)
        
        s4_ = self.s5_up_conv(s5)
        s4_ = s4_ + s4
        s4 = self.s4_fusion(s4_)
        
        s3_ = self.s4_up_conv(s4)
        s3_ = s3_ + s3
        s3 = self.s3_fusion(s3_)
        
        s2_ = self.s3_up_conv(s3)
        s2_ = s2_ + s2
        s2 = self.s2_fusion(s2_)
        
        s1_ = self.s2_up_conv(s2)
        s1_ = s1_ + s1
        s1 = self.s1_fusion(s1_)
        
        out = self.last_conv(s1)
        
        return out
    
    



























