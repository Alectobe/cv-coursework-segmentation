"""Attention Gate (собственная реализация).

Источник: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", 2018,
https://arxiv.org/abs/1804.03999 (раздел 2, рис. 2).

Идея: признаки skip-соединения x (высокое разрешение) умножаются на карту
коэффициентов alpha in [0, 1]. Карта считается по самому x и по сигналу g
с более грубого уровня декодера, который «знает» контекст и подсказывает,
какие области важны. Так в декодер меньше проходит фон (складки слизистой,
блики), похожий на полип по локальной текстуре.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """alpha = sigmoid(psi(relu(theta(x) + phi(g)))); выход: x * upsample(alpha)."""

    def __init__(self, skip_channels, gating_channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or max(skip_channels // 2, 1)
        # theta: свёртка 2x2 с шагом 2 приводит x к разрешению сигнала g (как в статье)
        self.theta = nn.Conv2d(skip_channels, inter_channels, kernel_size=2, stride=2, bias=False)
        # phi: проекция сигнала g в то же промежуточное пространство
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=True)
        # psi: одна карта внимания на все каналы x
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        # W: выходное преобразование после взвешивания (есть в эталонной реализации авторов)
        self.out = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
        )
        self.last_attention = None

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        # Для входов с нечётными размерами сетки могут разойтись на 1 пиксель
        if phi_g.shape[-2:] != theta_x.shape[-2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.psi(F.relu(theta_x + phi_g, inplace=True)))
        alpha = F.interpolate(alpha, size=x.shape[-2:], mode="bilinear", align_corners=False)
        # Сохраняем карту внимания для визуализации в отчёте
        self.last_attention = alpha.detach()
        return self.out(x * alpha)
