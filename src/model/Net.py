import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

"""
Following methods are created for ease of use.
    conv3x3: 2D convolutional layer with kernel_size=3x3, stride=1, padding=1
    conv1x1: 2D convolutional layer with kernel_size=1x1, stride=1
    conv2d_block: combination of following:
        Conv2d
        ReLU
        BatchNorm(optional. set via param 'bn'
"""
def conv3x3(in_channels, out_channels):
    return nn.Conv2d(kernel_size=3,in_channels=in_channels, out_channels=out_channels, stride=1,padding=(1,1))

def conv1x1(in_channels,out_channels):
    return nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1)

def conv2d_block(in_channels,out_channels,kernel_size=3,stride=1,padding=(1,1),bn=False):
    layers = []
    layers.append(nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,out_channels=out_channels, stride=stride,padding=padding))
    layers.append(nn.ReLU())
    if bn:
        layers.append(nn.BatchNorm2d(num_features=out_channels))
    return nn.Sequential(*layers)

class BottleNeck(nn.Module):
    """
    Bottleneck is essentially, a 2 times 3x3 conv2d blocks replaced by 1x1,3x3,1x1 blocks
    """
    def __init__(self, num_channels, width):
        # first 1x1 block
        super().__init__()
        self.conv1x1_1 = conv1x1(in_channels=num_channels, out_channels=width)
        self.bn1 = nn.BatchNorm2d(num_features=width)
        self.relu1 = nn.ReLU()

        # 3x3 block
        self.conv3x3 = conv3x3(width,width)
        self.bn2 = nn.BatchNorm2d(num_features=width)
        self.relu2 = nn.ReLU()

        # second 1x1 block
        self.conv1x1_2 = conv1x1(in_channels=width, out_channels=num_channels)
        self.bn3 = nn.BatchNorm2d(num_features=num_channels)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        # save input for residual block
        inp = x
        # forward through first 1x1 block
        x = self.conv1x1_1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # forward through 3x3 block
        x = self.conv3x3(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # forward through second 1x1 block
        x = self.conv1x1_2(x)
        x = self.bn3(x)
        x+=inp
        out = self.relu3(x)
        return out

class ResidualLayers(nn.Module):
    def __init__(self,bottleneck_count,channels):
        super(ResidualLayers, self).__init__()
        layers = [BottleNeck(num_channels=channels,width=channels) for _ in range(bottleneck_count)]
        self.res = nn.Sequential(*layers)

    def forward(self,x):
        out = self.res(x)
        return out


class Net(nn.Module):
    def __init__(self):
        super(DorukNet, self).__init__()
        self.conv_block_1 = conv2d_block(in_channels=3,out_channels=16)
        self.conv_block_2 = conv2d_block(in_channels=16,out_channels=64,bn=True)
        self.max_pool_1 = nn.MaxPool2d(stride=2,kernel_size=2)
        self.conv_block_3 = conv2d_block(in_channels=64, out_channels=256, bn=True)
        self.max_pool_2 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.res_layers = ResidualLayers(3,64)
        self.avg_pool_1 = nn.AvgPool2d(stride=2, kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(64*56*56,4096)
        self.drop_out_1 = nn.Dropout(p=0.2)
        self.fc_2 = nn.Linear(4096,1000)
        self.fc_3 = nn.Linear(1000,10)


    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.max_pool_1(x)
        res_inp = x
        x = self.res_layers(x)
        x+=res_inp
        x = self.avg_pool_1(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.drop_out_1(x)
        x = self.fc_2(x)
        out = self.fc_3(x)
        return out

