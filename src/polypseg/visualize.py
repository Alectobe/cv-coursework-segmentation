"""Наложение маски на снимок (только numpy: используется и в отчётах, и в сервисе)."""

import numpy as np


def error_map(gt, pred):
    """RGB-карта ошибок: TP - зелёный, FP - красный, FN - синий, TN - чёрный."""
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[(gt == 1) & (pred == 1)] = (0, 200, 0)
    out[(gt == 0) & (pred == 1)] = (230, 0, 0)
    out[(gt == 1) & (pred == 0)] = (0, 90, 255)
    return out


def mask_edges(mask):
    """Пиксели маски, у которых есть сосед вне маски (контур)."""
    m = np.pad(mask.astype(bool), 1, mode="edge")
    inner = m[1:-1, 1:-1]
    interior = m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return inner & ~interior


def overlay(image, mask, color=(0, 255, 0), alpha=0.45):
    """Полупрозрачная маска поверх снимка с контуром."""
    img = image.astype(np.float32).copy()
    m = mask.astype(bool)
    img[m] = (1 - alpha) * img[m] + alpha * np.array(color, dtype=np.float32)
    img[mask_edges(m)] = color
    return img.clip(0, 255).astype(np.uint8)
