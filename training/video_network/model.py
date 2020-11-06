import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape


class CNN3d(nn.Module):
    def __init__(self, t_dim=120, img_x=90, img_y=120, drop_p=0.2):
        super(CNN3d, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        #self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        #self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        #self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.k1, self.k2 = (3, 3, 3), (1, 3, 3)  # 3d kernel size
        #self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.s1, self.s2 = (1, 2, 2), (1, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding
        #self.pd1, self.pd2 = (1, 1, 1), (1, 1, 1)  # 3d padding
        
        self.ch3 = 64

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv1_outshape = tuple(x / 2 for x in self.conv1_outshape)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv2_outshape = tuple(x / 2 for x in self.conv2_outshape)
        
        assert np.all([divmod(x, 1)[1] == 0 for x in self.conv1_outshape]) and \
        np.all([divmod(x, 1)[1] == 0 for x in self.conv2_outshape]), \
        f'self.conv1_outshape={self.conv1_outshape}, self.conv2_outshape={self.conv2_outshape}'

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU() # inplace=True
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k2, stride=1,
                               padding=1)
        
        self.conv4 = nn.Conv3d(in_channels=self.ch3, out_channels=self.ch3, kernel_size=self.k2, stride=1,
                               padding=1)
        
        #self.fc1 = nn.Linear(self.ch2 * np.prod(self.conv2_outshape).astype(int),
        #                     self.fc_hidden1)  # fully connected hidden layer
        #self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        #self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes

        
        
    def forward(self, x):
        # Conv 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool(x)
        
        #x = self.conv3(x)
        x = self.relu(x)
        x = self.drop(x)
        #x = self.conv4(x)
        x = self.relu(x)
        # FC 1 and 2
        #x = x.view(x.size(0), -1)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.dropout(x, p=self.drop_p, training=self.training)
        #x = self.fc3(x)

        return x


class MultiResolution(nn.Module):
    
    def __init__(self, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, num_classes=2, **kwargs):
        super(MultiResolution, self).__init__()

        self.fovea = CNN3d(**kwargs)
        self.context = CNN3d(**kwargs)
        self.num_nets = 2
        self.drop_p = drop_p
        
        self.fc_hidden1 = fc_hidden1
        self.fc_hidden2 = fc_hidden2
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(self.num_nets * self.fovea.ch2 * np.prod(self.fovea.conv2_outshape).astype(int), fc_hidden1)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)
        
    def forward(self, x):
        x_c, x_f = x
        c_out = self.context(x_c)
        f_out = self.fovea(x_f)
        #print(x_c.shape)
        x = torch.cat([f_out, c_out], dim=1)
        #print(x.shape)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        
        return x
