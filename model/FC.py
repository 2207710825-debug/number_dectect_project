import torch
import torch.nn as nn


class CharacterNet(nn.Module):
    """
    根据架构图自定义的三隐藏层全连接神经网络 (MLP)
    """

    def __init__(self, num_classes=10):
        super(CharacterNet, self).__init__()

        # 隐藏层 1: 1024 -> 256
        self.fc1 = nn.Linear(1024, 256)#1024,256
        self.sigmoid1 = nn.Sigmoid()

        # 隐藏层 2: 256 -> 128
        self.fc2 = nn.Linear(256, 128)#256,128
        self.sigmoid2 = nn.Sigmoid()

        # 隐藏层 3: 128 -> 64
        self.fc3 = nn.Linear(128, 32)#128,64
        self.sigmoid3 = nn.Sigmoid()

        # 输出层: 64 -> 10
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        # 确保输入张量被展平为一维向量: [batch_size, 1024]
        x = x.view(x.size(0), -1)

        # 前向传播
        x = self.sigmoid1(self.fc1(x))
        x = self.sigmoid2(self.fc2(x))
        x = self.sigmoid3(self.fc3(x))
        x = self.out(x)

        return x