"""Датасет Kvasir-SEG: пары «снимок + маска полипа»."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from polypseg.data.transforms import load_mask, load_rgb, resize_image, to_tensor


def read_split(splits_dir, split):
    """Читает список идентификаторов (имён файлов без расширения) выборки."""
    path = Path(splits_dir) / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Нет файла разбиения {path}. Запустите scripts/prepare_data.py")
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"Файл разбиения {path} пустой")
    return ids


class KvasirSegDataset(Dataset):
    """Возвращает (изображение 3xSxS, маска 1xSxS со значениями {0, 1}, id)."""

    def __init__(self, root, ids, image_size, augment=None):
        self.root = Path(root)
        self.ids = list(ids)
        self.image_size = image_size
        self.augment = augment
        missing = [i for i in self.ids if not (self.root / "images" / f"{i}.jpg").exists()]
        if missing:
            raise FileNotFoundError(
                f"В {self.root / 'images'} нет {len(missing)} файлов из разбиения, например {missing[0]}.jpg"
            )

    def __len__(self):
        return len(self.ids)

    def image_path(self, sample_id):
        return self.root / "images" / f"{sample_id}.jpg"

    def mask_path(self, sample_id):
        return self.root / "masks" / f"{sample_id}.jpg"

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        image = np.asarray(resize_image(load_rgb(self.image_path(sample_id)), self.image_size))
        mask = load_mask(self.mask_path(sample_id), self.image_size)
        if self.augment is not None:
            out = self.augment(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        image_t = to_tensor(image)
        mask_t = torch.from_numpy(mask[None].astype(np.float32))
        return image_t, mask_t, sample_id
