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
import flowlib


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
    mean = img.new_tensor(VGG16_MEAN).view(-1, 1, 1)
    std = img.new_tensor(VGG16_STD).view(-1, 1, 1)
    if div:
        img = img.div_(255.0)
    else:
        img = img.add_(1).div_(2)
    return (img - mean) / std


normalize = transforms.Lambda(lambda x: normalizeVGG16(x))
normalize_after_reconet = transforms.Lambda(lambda x: normalizeVGG16(x, div=False))

transform2 = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (1-x).mul(255)),
                normalize
                ])


def gram_matrix(inp):
    # print(input.size())
    a, b, c, d = inp.size()
    #print(inp.permute(0, 2, 3, 1))
    features = inp.view(a * b, c * d)
    #print(features)
    G = torch.mm(features, features.t())
    #print(G)
    #print(torch.mm(features[:1], features[:1].t()))
    #print(G.mean())
    return G.div(a * b * c * d)

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



