"""Загрузка весов и предсказание внутри сервиса."""

import base64
import hashlib
import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from polypseg.inference import load_checkpoint, predict_proba
from polypseg.visualize import overlay


def png_bytes(array):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def b64(data):
    return base64.b64encode(data).decode("ascii")


class Predictor:
    def __init__(self, checkpoint_path, device=None):
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Нет файла весов {path}. Получите веса: docker compose run --rm weights (см. README)"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.meta = load_checkpoint(path, device=self.device)
        self.image_size = int(self.meta["image_size"])
        self.threshold = float(self.meta["threshold"])
        clearml = self.meta.get("clearml") or {}
        self.model_info = {
            "experiment": self.meta["experiment"],
            "epoch": self.meta.get("epoch"),
            "val_dice": (self.meta.get("val_metrics") or {}).get("dice"),
            "image_size": self.image_size,
            "default_threshold": self.threshold,
            "clearml_task_id": clearml.get("task_id"),
            "clearml_task_url": clearml.get("task_url"),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        # Прогрев: первый проход медленнее, и /health должен отвечать ok только когда модель реально работает
        self.predict(Image.new("RGB", (self.image_size, self.image_size)))

    def predict(self, image, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        t0 = time.perf_counter()
        prob = predict_proba(self.model, image, self.image_size, self.device)
        mask = prob > threshold
        elapsed_ms = 1000 * (time.perf_counter() - t0)

        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None
        return {
            "mask": mask,
            "threshold": threshold,
            "polyp_found": bool(mask.any()),
            "polyp_area_ratio": float(mask.mean()),
            "bbox": bbox,
            "mean_confidence": float(prob[mask].mean()) if mask.any() else 0.0,
            "max_probability": float(prob.max()),
            "inference_ms": round(elapsed_ms, 1),
        }

    @staticmethod
    def mask_png(mask):
        return png_bytes(mask.astype(np.uint8) * 255)

    @staticmethod
    def overlay_png(image, mask):
        return png_bytes(overlay(np.asarray(image.convert("RGB")), mask, color=(0, 255, 0)))
