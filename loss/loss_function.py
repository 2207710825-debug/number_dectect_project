# -*- coding: utf-8 -*-
'''
@Author   ：shenwenjun
@Date     ：2026/9/14 17:36
@Describe ：定义损失函数，通过调用损失函数
'''
import torch
import torch.nn as nn

class CustomCrossEntropyLoss(nn.Module):
    """
    自定义损失函数类：继承 nn.Module，对内部 CrossEntropyLoss 进行封装扩展
    """
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        # 计算基础交叉熵损失
        loss = self.criterion(predictions, targets)
        return loss