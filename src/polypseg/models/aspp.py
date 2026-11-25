"""Atrous Spatial Pyramid Pooling (собственная реализация).

Источник: Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation"
(DeepLabv3), 2017, https://arxiv.org/abs/1706.05587, раздел 3.3.

Блок ставится в bottleneck U-Net (самый глубокий уровень энкодера, 1/32 входа).
Параллельные свёртки с разной dilation и глобальный pooling собирают контекст
разных масштабов: полипы в Kvasir-SEG занимают от 0.5% до 80% кадра.

Отступление от статьи: dilation (1, 2, 4, 6) вместо (1, 6, 12, 18).
На входе 352x352 карта bottleneck имеет размер 11x11, и при dilation 12-18
ядро 3x3 почти целиком попадает в паддинг.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch, out_ch, kernel_size, dilation=1):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ASPPPooling(nn.Module):
    """Ветка image-level features: глобальный pooling -> 1x1 conv -> растяжение обратно."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = _conv_bn_relu(in_ch, out_ch, 1)

    def forward(self, x):
        y = self.proj(self.pool(x))
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Вход и выход имеют одинаковое число каналов, поэтому блок встаёт на место
    центрального блока декодера без изменения остальных слоёв."""

    def __init__(self, in_channels, out_channels, rates=(1, 2, 4, 6), branch_channels=128, dropout=0.1):
        super().__init__()
        branches = []
        for rate in rates:
            if rate == 1:
                branches.append(_conv_bn_relu(in_channels, branch_channels, 1))
            else:
                branches.append(_conv_bn_relu(in_channels, branch_channels, 3, dilation=rate))
        branches.append(ASPPPooling(in_channels, branch_channels))
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            _conv_bn_relu(branch_channels * len(branches), out_channels, 1),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))
