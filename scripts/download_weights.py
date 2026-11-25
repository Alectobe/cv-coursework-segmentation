"""Получение итоговых весов модели для сервиса без повторного обучения.

Источники (в порядке приоритета):
1. --clearml-model-id / переменная CLEARML_MODEL_ID - модель из реестра ClearML;
2. --url / переменная WEIGHTS_URL - файл из GitHub Release (по умолчанию).

Файл сохраняется в artifacts/final/model.pt и проверяется: загружается ли
чекпоинт и из какого эксперимента ClearML получены веса.
"""

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import torch

REQUIRED_KEYS = ("state_dict", "config", "experiment", "image_size", "threshold")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path):
    """Проверяет чекпоинт и печатает, откуда он."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ValueError(f"{path} не является чекпоинтом проекта: нет ключей {missing}")
    clearml = ckpt.get("clearml") or {}
    print(f"[weights] файл: {path} ({Path(path).stat().st_size / 2**20:.1f} МБ, sha256 {sha256(path)[:16]}...)")
    print(f"[weights] эксперимент: {ckpt['experiment']}, эпоха {ckpt.get('epoch')}, val Dice {ckpt.get('val_metrics', {}).get('dice')}")
    print(f"[weights] ClearML task: {clearml.get('task_id')}  {clearml.get('task_url') or ''}")


def from_url(url, dst):
    print(f"[weights] скачиваю {url}")
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dst)


def from_clearml(model_id, dst):
    from clearml import InputModel

    print(f"[weights] скачиваю модель {model_id} из ClearML")
    local = InputModel(model_id=model_id).get_local_copy()
    if not local:
        raise RuntimeError(f"ClearML не вернул файл модели {model_id}")
    shutil.copyfile(local, dst)


def main():
    p = argparse.ArgumentParser(description="Скачивание итоговых весов")
    p.add_argument("--url", default=os.environ.get("WEIGHTS_URL"))
    p.add_argument("--clearml-model-id", default=os.environ.get("CLEARML_MODEL_ID"))
    p.add_argument("--sha256", default=os.environ.get("WEIGHTS_SHA256"), help="ожидаемый SHA-256 файла (необязательно)")
    p.add_argument("--out", default="artifacts/final/model.pt")
    p.add_argument("--force", action="store_true", help="скачать заново, даже если файл уже есть")
    args = p.parse_args()

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not args.force:
        print("[weights] файл уже есть, повторно не скачиваю (используйте --force)")
        describe(dst)
        return

    try:
        if args.clearml_model_id:
            from_clearml(args.clearml_model_id, dst)
        elif args.url:
            from_url(args.url, dst)
        else:
            print("[weights] не задан источник: укажите WEIGHTS_URL или CLEARML_MODEL_ID в .env", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - показываем причину и выходим с кодом ошибки
        print(f"[weights] не удалось получить веса: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.sha256 and sha256(dst) != args.sha256:
        dst.unlink()
        print("[weights] SHA-256 не совпадает с ожидаемым, файл удалён", file=sys.stderr)
        sys.exit(1)
    describe(dst)


if __name__ == "__main__":
    main()
