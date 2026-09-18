# -*- coding: utf-8 -*-
'''
@Author   ：shenwenjun
@Date     ：2026/9/17 22:18
@Describe ：
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
class CharacterNet(nn.Module):

    def __init__(self, num_classes=10):
        super(CharacterNet, self).__init__()

        # 特征提取提取层
        self.features = nn.Sequential(
            # 第一层卷积：输入 3 通道(RGB)，输出 32 通道，卷积核 3x3，padding 1
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样：图像宽高减半
            # 第二层卷积：输入 32 通道，输出 64 通道
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样
            # 第三层卷积：输入 64 通道，输出 128 通道
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 下采样
        )

        # 分类器层
        # 假设输入图像大小为 32x32，经过 3 次 MaxPool(2x2) 后，尺寸变为 4x4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # 防止过拟合
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x