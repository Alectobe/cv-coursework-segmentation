"""Метрики бинарной сегментации.

Все метрики считаются по каждому изображению отдельно и затем усредняются
(mDice, mIoU - как принято в работах по сегментации полипов). Так маленький
полип весит столько же, сколько большой; при глобальном подсчёте по пикселям
ошибки на маленьких полипах «растворялись» бы в больших.
"""

import torch


def confusion_counts(pred, target):
    """pred, target: бинарные тензоры Nx1xHxW -> TP, FP, FN по каждому изображению (N,)."""
    pred = pred.bool().flatten(1)
    target = target.bool().flatten(1)
    tp = (pred & target).sum(1).double()
    fp = (pred & ~target).sum(1).double()
    fn = (~pred & target).sum(1).double()
    return tp, fp, fn


def scores_from_counts(tp, fp, fn):
    """Dice, IoU, precision, recall по счётчикам.

    Пограничные случаи: если и маска, и предсказание пустые - Dice = IoU = 1;
    если предсказание пустое - precision = 1 только при пустой маске, иначе 0.
    """
    one = torch.ones_like(tp)
    zero = torch.zeros_like(tp)
    both_empty = (tp + fp + fn) == 0

    dice = torch.where(both_empty, one, 2 * tp / (2 * tp + fp + fn).clamp(min=1))
    iou = torch.where(both_empty, one, tp / (tp + fp + fn).clamp(min=1))
    precision = torch.where((tp + fp) == 0, torch.where(fn == 0, one, zero), tp / (tp + fp).clamp(min=1))
    recall = torch.where((tp + fn) == 0, one, tp / (tp + fn).clamp(min=1))
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


class SegmentationMeter:
    """Накапливает поизображенческие метрики за эпоху."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.values = {"dice": [], "iou": [], "precision": [], "recall": []}

    @torch.no_grad()
    def update(self, logits, targets):
        pred = torch.sigmoid(logits.float()) > self.threshold
        scores = scores_from_counts(*confusion_counts(pred, targets > 0.5))
        for name, value in scores.items():
            self.values[name].append(value.cpu())

    def compute(self):
        if not self.values["dice"]:
            raise RuntimeError("SegmentationMeter.compute() вызван без данных")
        return {name: torch.cat(vals).mean().item() for name, vals in self.values.items()}
