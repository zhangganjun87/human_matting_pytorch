#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhangganjun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from Tnet import T_mv2_unet
from Mnet import M_net

class net(nn.Module):
    
    def __init__(self):
        super(net, self).__init__()
        self.t_net = T_mv2_unet()
        self.m_net = M_net()
        
    def forward(self, input):
        
        # trimap
        trimap = self.t_net(input)
        trimap_softmax = F.softmax(trimap, dim=1)
        
        bg, fg, unsure = torch.split(trimap_softmax, 1, dim=1)
        
        #concat input and trimap
        m_net_input = torch.cat((input, trimap_softmax), 1)
        
        #matting
        alpha_r = self.m_net(m_net_input)
        #alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r
        
        return trimap, alpha_p

































