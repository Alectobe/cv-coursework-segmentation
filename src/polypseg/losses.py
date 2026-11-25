"""Функции потерь для бинарной сегментации.

BCE - функция потерь исходной модели (стандартная из PyTorch).
Dice, Focal и их сумма реализованы самостоятельно - это модификация
«реализовать и подключить другую функцию потерь».

Зачем Dice + Focal: полип в среднем занимает около 11% кадра (медиана по Kvasir-SEG),
поэтому в BCE доминируют легко классифицируемые пиксели фона.
- Dice оптимизирует перекрытие маски напрямую и не зависит от доли фона;
- Focal (Lin et al., 2017, https://arxiv.org/abs/1708.02002) уменьшает вклад
  уверенно классифицированных пикселей и концентрирует обучение на сложных:
  границах полипа, бликах, складках.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """1 - Dice, посчитанный по вероятностям; усредняется по изображениям батча."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits.float()).flatten(1)
        targets = targets.float().flatten(1)
        intersection = (probs * targets).sum(1)
        denominator = probs.sum(1) + targets.sum(1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BinaryFocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    alpha=None отключает балансировку классов: дисбаланс уже учитывает Dice.
    """

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t - вероятность правильного класса; exp(-bce) численно устойчивее, чем через sigmoid
        p_t = torch.exp(-bce)
        loss = (1.0 - p_t) ** self.gamma * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss.mean()


class DiceFocalLoss(nn.Module):
    """Взвешенная сумма Dice и Focal."""

    def __init__(self, dice_weight=1.0, focal_weight=1.0, gamma=2.0, alpha=None):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = SoftDiceLoss()
        self.focal = BinaryFocalLoss(gamma=gamma, alpha=alpha)

    def forward(self, logits, targets):
        return self.dice_weight * self.dice(logits, targets) + self.focal_weight * self.focal(logits, targets)


class BCELoss(nn.Module):
    """BCEWithLogits с приведением к float32 (устойчиво при AMP)."""

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits.float(), targets.float())


def build_loss(loss_cfg):
    name = loss_cfg["name"]
    if name == "bce":
        return BCELoss()
    if name == "dice":
        return SoftDiceLoss()
    if name == "focal":
        return BinaryFocalLoss(gamma=loss_cfg.get("focal_gamma", 2.0), alpha=loss_cfg.get("focal_alpha"))
    if name == "dice_focal":
        return DiceFocalLoss(
            dice_weight=loss_cfg.get("dice_weight", 1.0),
            focal_weight=loss_cfg.get("focal_weight", 1.0),
            gamma=loss_cfg.get("focal_gamma", 2.0),
            alpha=loss_cfg.get("focal_alpha"),
        )
    raise ValueError(f"Неизвестная функция потерь: {name}")
