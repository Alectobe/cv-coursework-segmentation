"""Загрузка чекпоинта и предсказание маски для одного изображения.

Используется и в оценке на тесте (evaluate.py), и в сервисе инференса, поэтому
оба получают одинаковый препроцессинг и одинаковую постобработку.
"""

import torch
import torch.nn.functional as F

from polypseg.data.transforms import preprocess_image
from polypseg.models.factory import build_model

REQUIRED_KEYS = ("state_dict", "config", "experiment", "image_size", "threshold")


def load_checkpoint(path, device="cpu"):
    """Возвращает (модель в режиме eval, словарь чекпоинта без весов)."""
    # weights_only=True: из файла загружаются только тензоры и простые типы, без произвольного кода
    ckpt = torch.load(path, map_location=device, weights_only=True)
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"Файл {path} не похож на чекпоинт проекта: нет ключей {missing}")
    model = build_model(ckpt["config"]["model"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta


@torch.inference_mode()
def predict_proba(model, image, image_size, device="cpu"):
    """PIL-изображение -> карта вероятностей полипа HxW в исходном разрешении (numpy float32)."""
    width, height = image.size
    x = preprocess_image(image, image_size).to(device)
    logits = model(x)
    # Логиты растягиваются до исходного размера снимка, затем применяется сигмоида
    logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
    return torch.sigmoid(logits)[0, 0].float().cpu().numpy()
