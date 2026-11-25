"""Загрузка конфигурации эксперимента и фиксация случайности."""

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

REQUIRED_SECTIONS = ("experiment", "data", "model", "loss", "train", "tracker", "output")


def load_config(path):
    """Читает YAML-конфиг и проверяет наличие обязательных секций."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"В конфиге {path} нет секций: {', '.join(missing)}")
    cfg["experiment"].setdefault("tags", [])
    cfg["_source"] = str(path)
    return cfg


def save_config(cfg, path):
    """Сохраняет конфиг рядом с результатами эксперимента."""
    data = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def set_seed(seed):
    """Фиксирует seed во всех используемых генераторах случайных чисел."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def file_sha256(path):
    """SHA-256 файла: используется для фиксации версии разбиения выборок."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
