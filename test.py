#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhangganjun
"""

import time
import cv2
import torch
import argparse
import numpy as np
import os
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='human_matting')
parser.add_argument('--model', default='./', help='preTrained model')
parser.add_argument('--size', type=int,default=256)
parser.add_argument('--without_gpu', action='store_true', default=False)

args = parser.parse_args()

torch.set_grad_enabled(False)

if args.without_gpu:
    print('use cpu')
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print('available gpu number is {}'.format(n_gpu))
        device = torch.device('cuda:0,1')
        
def load_model(args):
    print('Loading model from {}...'.format(args.model))
    if args.without_gpu:
        myModel = torch.load(args.model, map_location=lambda storage, loc: storage)
    else:
        myModel = torch.load(args.model)
        
    myModel.eval()
    myModel.to(device)
    
    return myModel

def seg_process(args, image, net):
    origin_h, origin_w, c = image.shape
    image_resize = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
    image_resize = (image_resize - (104., 112., 121.)) /255.0
    
    tensor_4D = torch.FloatTensor(1, 3, args.size, args.size)
    tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
    inputs = tensor_4D.to(device)
    
    t0 = time.time()
    
    trimap, alpha = net(inputs)
    
    print(time.time() - t0)
    
    if args.without_gpu:
        alpha_np = alpha[0,0,:,:].data.numpy()
    else:
        alpha_np = alpha[0,0,:,:].cpu().data.numpy()
        
    alpha_np = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
    
    fg = np.multiply(alpha_np[..., np.newaxis], image)
    
    bg = image
    bg_gray = np.multiply(1-alpha_np[..., np.newaxis], image)
    bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)
    
    bg[:,:,0] = bg_gray
    bg[:,:,1] = bg_gray
    bg[:,:,2] = bg_gray
    
    out = fg + bg
    out[out<0] = 0
    out[out>255] = 255
    out = out.astype(np.uint8)
    
    return out

def camera_seg(args, net):

    videoCapture = cv2.VideoCapture(0)

    while(1):
        # get a frame
        ret, frame = videoCapture.read()
        frame = cv2.flip(frame,1)
        frame_seg = seg_process(args, frame, net)


        # show a frame
        cv2.imshow("capture", frame_seg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    videoCapture.release()
    
def main(args):
    image_path = './'
    image = cv2.imread(image_path)
    myModel = load_model(args)
    seg_process(args, image, myModel)
    
if __name__=='__main__':
    main(args)































