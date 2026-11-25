"""Копирует несколько снимков из ТЕСТОВОЙ выборки в examples/ для демонстрации сервиса.

    python scripts/export_examples.py --n 6

Результат: examples/polyp_01.jpg ... polyp_06.jpg (от самого маленького полипа к самому крупному),
рядом эталонные маски *_gt_mask.jpg и index.csv с исходными id снимков.

Тестовые снимки не использовались ни при обучении, ни при выборе модели.
Выбираются полипы разного размера: от самых маленьких до самых больших.
Снимки в git не хранятся (лицензия Kvasir-SEG разрешает использование в
исследовательских и учебных целях, но не перераспространение).
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from polypseg.data.dataset import read_split
from polypseg.data.transforms import load_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/Kvasir-SEG")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--out", default="examples")
    p.add_argument("--n", type=int, default=6)
    args = p.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = read_split(args.splits_dir, "test")
    areas = sorted((float(load_mask(root / "masks" / f"{i}.jpg").mean()), i) for i in ids)
    picks = [areas[k] for k in np.linspace(0, len(areas) - 1, args.n).round().astype(int)]
    rows = ["file,id,polyp_area_percent"]
    for k, (area, sample_id) in enumerate(picks, 1):
        # Короткие имена файлов, чтобы команды из README работали без подстановки id
        dst = out / f"polyp_{k:02d}.jpg"
        shutil.copyfile(root / "images" / f"{sample_id}.jpg", dst)
        shutil.copyfile(root / "masks" / f"{sample_id}.jpg", out / f"polyp_{k:02d}_gt_mask.jpg")
        rows.append(f"{dst.name},{sample_id},{100 * area:.2f}")
        print(f"[examples] {dst} <- {sample_id} (полип {100 * area:.1f}% кадра)")
    (out / "index.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
