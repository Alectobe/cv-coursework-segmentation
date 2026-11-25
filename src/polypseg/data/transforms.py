"""Препроцессинг и аугментации.

Препроцессинг для val/test и для сервиса один и тот же (PIL + numpy), поэтому
модель в сервисе получает данные ровно в том виде, в каком их видела при валидации.
Аугментации (albumentations) применяются только при обучении и уже после ресайза.
"""

import os

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Порог бинаризации масок: маски в Kvasir-SEG хранятся в JPEG
MASK_THRESHOLD = 127


def load_rgb(path):
    """Открывает изображение и приводит к RGB."""
    with Image.open(path) as img:
        return img.convert("RGB")


def resize_image(img, size):
    """Ресайз изображения до size x size (билинейная интерполяция)."""
    return img.resize((size, size), Image.BILINEAR)


def load_mask(path, size=None):
    """Загружает маску как массив {0, 1}.

    При ресайзе маска сначала интерполируется в оттенках серого, затем
    бинаризуется: так граница получается ровнее, чем при nearest.
    """
    with Image.open(path) as m:
        mask = m.convert("L")
    if size is not None:
        mask = mask.resize((size, size), Image.BILINEAR)
    return (np.asarray(mask) > MASK_THRESHOLD).astype(np.uint8)


def to_tensor(image_np):
    """uint8 HxWx3 -> нормализованный float-тензор 3xHxW."""
    x = image_np.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def preprocess_image(img, size):
    """PIL-изображение -> батч 1x3xSxS для модели."""
    return to_tensor(np.asarray(resize_image(img.convert("RGB"), size))).unsqueeze(0)


def build_train_augment():
    """Аугментации для обучения.

    У эндоскопического снимка нет «правильной» ориентации, поэтому допустимы
    отражения и повороты на 90 градусов. Цветовые искажения небольшие:
    цвет слизистой и полипа несёт информацию.
    """
    # Без этого albumentations при импорте ходит в сеть проверять свою версию
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-20, 20), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02, p=0.5),
        ]
    )
