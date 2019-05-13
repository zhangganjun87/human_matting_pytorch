#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhangganjun
"""

import cv2
import os
import random
import numpy as np
import torch
import torch.utils.data as data


def read_files(data_dir, file_name={}):
    image_name = os.path.join(data_dir, 'image', file_name['image'])
    alpha_name = os.path.join(data_dir, 'alpha', file_name['alpha'])
    
    image = cv2.imread(image_name)
    alpha = cv2.imread(alpha_name)
    
#    image_ = cv2.resize(image,(512,800))
#    alpha_ = cv2.resize(alpha,(512,800))
    image_ = cv2.resize(image,(128,128))
    alpha_ = cv2.resize(alpha,(128,128))
    
    return image_, alpha_

def random_scale_and_creat_patch(image, alpha, patch_size):
    if random.random() < 0.5:
        h, w, c = image.shape
        scale = 0.75 + 0.5*random.random()
        image = cv2.resize(image, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_CUBIC)
        alpha = cv2.resize(alpha, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_CUBIC)
        
    if random.random() < 0.5:
        h, w, c = image.shape
        if h > patch_size and w > patch_size:
            x = random.randrange(0, w-patch_size)
            y = random.randrange(0, h-patch_size)
            image = image[y:y+patch_size, x:x+patch_size, :]
            alpha = alpha[y:y+patch_size, x:x+patch_size, :]
        else:
            image = cv2.resize(image, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
            alpha = cv2.resize(alpha, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
            
    else:
        image = cv2.resize(image, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
        alpha = cv2.resize(alpha, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
        
    return image, alpha

def random_flip(image, alpha):
    if random.random() < 0.5:
        image = cv2.flip(image, 0)
        alpha = cv2.flip(alpha, 0)
        
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        alpha = cv2.flip(alpha, 1)
        
    return image, alpha

def np2Tensor(array):
    ts = (2, 0, 1)
    tensor = torch.FloatTensor(array.transpose(ts).astype(float))
    return tensor

class human_matting_data(data.Dataset):
    
    def __init__(self, root_dir,patch_size):
        super().__init__()
        self.data_root = root_dir
        self.patch_size = patch_size
        self.imgDir = self.data_root + 'image/'
        self.imgID = os.listdir(self.imgDir)    #['dataset.py']
        # need fileter dir
        self.num = len(self.imgID)
        print('Dataset : file number %d' % self.num)
        
    def __getitem__(self, index):
        image, alpha = read_files(self.data_root, file_name={
                'image':self.imgID[index].strip(),
                'alpha':self.imgID[index].strip().split('.')[0]+'.png',})
        
        #normalize
        image = (image.astype(np.float32) - (114., 121., 134.,)) / 255.0
        alpha = alpha.astype(np.float32) / 255.0
        
        #to tensor
        image = np2Tensor(image)
        alpha = np2Tensor(alpha)
        
        alpha = alpha[0,:,:].unsqueeze_(0)
        sample = {'image':image, 'alpha':alpha}
        
        return sample
    
    def __len__(self):
        return self.num
        
        






























