import torch
import torch.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import flowlib


device = 'cuda'
ALPHA = 1e13 #previously 12, 2e10 // 1e4
BETA  = 1e10 #1e6 #11, // 1e5
GAMMA = 3e-2 #previously -3 // 1e-5
LAMBDA_O = 1e6 # // 2e5
LAMBDA_F = 1e4 # // e5
IMG_SIZE = (640, 360)
VGG16_MEAN = [0.485, 0.456, 0.406]
VGG16_STD = [0.229, 0.224, 0.225]

def normalizeVGG16(img, div=True):
    if len(img.shape) == 3:
        pix_count = np.prod(img.shape[:2])
        shape = img.shape
    elif len(img.shape) == 4:
        pix_count = np.prod(img.shape[1:3])
        shape = img.shape[1:]
    mean = np.array(VGG16_MEAN * pix_count).reshape(shape)
    std = np.array(VGG16_STD * pix_count).reshape(shape)
    if div:
        #img = img.div_(255.0)
        img = img / 255.0
    else:
        pass
        #img = img.add_(1).div_(2)
    return (img - mean) / std

normalize = lambda x: normalizeVGG16(x)
normalize_after_reconet = lambda x: normalizeVGG16(x, div=False)

#normalize = transforms.Lambda(lambda x: normalizeVGG16(x))
#normalize_after_reconet = transforms.Lambda(lambda x: normalizeVGG16(x, div=False))


def gram_matrix(inp):
    print(inp.shape)
    #inp = np.array(inp)
    a, b, c, d = inp.shape
    #features = inp.reshape(a * b, c * d)
    shape = (a * d, b * c)
    features = tf.reshape(inp, shape)
    #G = np.dot(features, features.transpose())
    G = tf.linalg.matmul(features, features, transpose_b=True)
    return G / (a * b * c * d)

def warp(x, flo, device):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
        
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask



