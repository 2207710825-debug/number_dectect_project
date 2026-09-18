# -*- coding: utf-8 -*-
'''
@Author   ：shenwenjun
@Date     ：2026/9/16 23:21
@Describe ：RNN
'''
import torch
import torch.nn as nn


class CharacterNet(nn.Module):
    """
    基于单层/多层 RNN 的手写数字识别网络
    针对 32x32 单通道图像输入
    """

    def __init__(self, num_classes=10, hidden_size=128, num_layers=2):
        super(CharacterNet, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN 层：
        # input_size=32：每行 32 个像素作为特征向量
        # hidden_size=128：RNN 隐藏状态的维度
        # num_layers=2：堆叠 2 层 RNN，增加网络表达能力
        # batch_first=True：输入的张量形状为 [batch_size, seq_len, input_size]
        self.rnn = nn.RNN(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu'  # 使用 ReLU 激活函数（比 tanh 更不容易梯度消失）
        )

        # 全连接输出层：将 RNN 最后一个时间步的隐状态映射到分类类别数 (10)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 1. 调整输入维度
        # 原始输入形状: [batch_size, 1, 32, 32] 或 [batch_size, 32, 32]
        # 转换为 RNN 要求的: [batch_size, seq_len=32, input_size=32]
        x = x.view(x.size(0), 32, 32)

        # 2. 前向传播经过 RNN
        # out 形状: [batch_size, seq_len(32), hidden_size(128)]
        # _ (h_n) 形状: [num_layers, batch_size, hidden_size]
        out, _ = self.rnn(x)

        # 3. 取最后一个时间步（最后一个序列时刻，即第 32 行）的输出用于分类
        out = out[:, -1, :]  # 形状变为 [batch_size, hidden_size]

        # 4. 通过全连接层得出最终类别 Logits
        x = self.out(out)

        return x