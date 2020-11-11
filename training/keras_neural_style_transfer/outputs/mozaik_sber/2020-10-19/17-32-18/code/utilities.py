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
from tensorflow.keras.applications.vgg16 import preprocess_input
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import shutil
import hydra
import os

IMG_SIZE = (600, 600)
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
        img = img / 255
    else:
        img = (img + 1) / 2

    return (img - mean) / std


def normalize(x): return normalizeVGG16(x)


def normalize_after_reconet(x): return normalizeVGG16(x, div=False)


def gram_matrix(inp):
    a, b, c, d = inp.shape
    G = tf.linalg.einsum('bijc,bijd->bcd', inp, inp)

    return G / (a * b * c * d)


def save_code() -> None:
    print(hydra.utils.get_original_cwd())
    print(os.getcwd())
    os.makedirs(os.path.join(os.getcwd(), "code"), exist_ok=True)
    for filename in ["train.py", "network.py", "model.py", "utilities.py"]:
        shutil.copy2(
            os.path.join(hydra.utils.get_original_cwd(), filename),
            os.path.join(os.getcwd(), "code", filename),
        )
