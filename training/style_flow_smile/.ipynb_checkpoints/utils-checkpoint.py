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
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2, ToTensor
import torchvision.transforms as T

import numpy as np


device = 'cuda'
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

def load_state_dict(model, model_path, device, source=''):
    
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    if source == 'jester':
        pretrained_renamed_dict = {k.replace('module.', ''): v for k, v in pretrained_dict['state_dict'].items()}
        pretrained_dict = pretrained_renamed_dict

    model_dict_new = model_dict.copy()
    counter = 0
    for k, v in pretrained_dict.items():
        if k in model_dict_new and np.all(v.size() == model_dict_new[k].size()):
            model_dict_new.update({k: v})
            counter += 1
    print(f'Loaded: {counter}/{len(model_dict)}')

    model.load_state_dict(model_dict_new)
    return model

def additional_augmenation(image, pp, seed=0, mode="torch"):
    random.seed(seed)
    compose_list = [
        #A.Resize(*size),
        # A.RandomCrop(120, 120),
        A.Blur(blur_limit=(4, 6), p=pp['Blur']),
        A.RandomBrightness(p=pp['RandomBrightness'], limit=(-0.3, 0.3)),
        A.JpegCompression(quality_lower=35, quality_upper=70, p=pp['JpegCompression']),
        A.GaussNoise(var_limit=1000, p=pp['GaussNoise']),
        # A.RandomSunFlare(p=p_small),
        # A.Downscale(p=p_small),
        # A.CLAHE(p=0.05),
        # A.RandomContrast(p=0.05),
        # A.RandomBrightness(p=0.05),
        A.HorizontalFlip(p=pp['HorizontalFlip']),
        # A.VerticalFlip(),
        # A.RandomRotate90(),
        A.ShiftScaleRotate(
            shift_limit=0.12, scale_limit=0.12, rotate_limit=5, p=pp['ShiftScaleRotate']
        ),
        # A.OpticalDistortion(p=1),
        # A.GridDistortion(p=1, num_steps=12, distort_limit=0.7),
        # A.ChannelShuffle(p=1),
        # A.HueSaturationValue(p=0.05),
        # A.ElasticTransform(),
        A.ToGray(p=pp['ToGray']),
        # A.JpegCompression(p=0.05),
        # A.MedianBlur(p=0.05),
        # A.Cutout(p=0.05),
        # A.RGBShift(p=p_small),
        #A.GaussNoise(var_limit=(0, 50), p=0.05),
        # A.Normalize(),
    ]
    if mode == "torch":
        compose_list.append(ToTensorV2())
    comp = A.Compose(compose_list, p=1)

    image = np.array(image).astype("uint8")
    image = comp(image=image)["image"]
    image = image.div(255.0).mul(2).sub(1)
    return image

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
