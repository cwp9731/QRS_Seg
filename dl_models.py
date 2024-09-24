# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :dl_models
# @Time      :2024/8/15 23:38
# @Author    :Chen
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv1d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm1d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv1d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm1d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv1d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv1d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 主干网络
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # 4次下采样
        self.C1 = Conv(1, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)

        self.U1 = UpSampling(512)
        self.C5 = Conv(512, 256)
        self.U2 = UpSampling(256)
        self.C6 = Conv(256, 128)
        self.U3 = UpSampling(128)
        self.C7 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv1d(64, 1, 3, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        Y1 = self.C4(self.D3(R3))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C5(self.U1(Y1, R3))
        O2 = self.C6(self.U2(O1, R2))
        O3 = self.C7(self.U3(O2, R1))

        return self.Th(self.pred(O3))

