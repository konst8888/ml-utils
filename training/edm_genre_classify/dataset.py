import os
import cv2
import sys
import random
import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image


class AudioDataset(Dataset):

    def __init__(
        self,
        data,
        idxs,
        classes,
        transform=None
    ):
        self.data = data
        self.transform = transform
        self.classes = classes
        self.idx2split_idx = {i: idx for i, idx in enumerate(idxs)}

    def __len__(self):
        return len(self.idx2split_idx)

    def __getitem__(self, idx):

        split_idx = self.idx2split_idx[idx]
        img = self.data['x'][split_idx]
        label = self.data['y'][split_idx]
        label = self.classes.index(label)
        if self.transform is not None:
            img = self.transform(img)

        #label = torch.LongTensor(label)
        return img, label
