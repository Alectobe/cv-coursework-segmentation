"""Проверка данных, поиск дубликатов и разбиение на train / val / test.

    python scripts/prepare_data.py            # создать разбиение или проверить существующее
    python scripts/prepare_data.py --force    # пересоздать разбиение

Состав выборок хранится в git (data/splits/*.txt). Если файлы разбиения уже есть,
скрипт их НЕ перезаписывает, а проверяет: все снимки на месте, выборки не
пересекаются, группы дубликатов не разорваны между выборками.

Дубликаты. Глобальный pHash на этих данных бесполезен: почти все кадры имеют
одинаковую восьмиугольную рамку эндоскопа и чёрный прямоугольник в углу, поэтому
совершенно разные снимки дают расстояние 2-8. Используется pHash 16x16 (256 бит)
по центральной части кадра (60% по ширине и высоте). На Kvasir-SEG расстояния
распределены так: 3 пары <= 6 (побайтовые копии), 1 пара = 48 (соседние кадры
одной сцены), дальше разрыв до 62+ (разные снимки, проверено визуально).
Порог 56 отделяет копии и почти-копии от разных снимков. Найденные пары
не удаляются, а объединяются в группы, и каждая группа целиком попадает в одну выборку.

Разбиение стратифицировано по размеру полипа (доля площади кадра), чтобы
в маленьких val и test (по 100 снимков) распределение размеров было как в train.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from polypseg.config import file_sha256  # noqa: E402
from polypseg.data.source import ARCHIVE_NAME, KVASIR_SHA256, KVASIR_URL  # noqa: E402
from polypseg.data.transforms import load_mask  # noqa: E402

SPLITS = ("train", "val", "test")
SIZE_BINS = ((0.0, 0.05), (0.05, 0.20), (0.20, 1.01))
CENTER_CROP = 0.2  # доля, отрезаемая с каждой стороны перед хэшированием


def find_dataset_root(raw_dir):
    """Ищет каталог, внутри которого есть images/ и masks/."""
    raw_dir = Path(raw_dir)
    for candidate in [raw_dir, *sorted(p for p in raw_dir.rglob("*") if p.is_dir())]:
        if (candidate / "images").is_dir() and (candidate / "masks").is_dir():
            return candidate
    raise FileNotFoundError(f"В {raw_dir} не найден каталог с images/ и masks/. Сначала запустите download_data.py")


def center_phash(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        crop = img.crop((int(w * CENTER_CROP), int(h * CENTER_CROP), int(w * (1 - CENTER_CROP)), int(h * (1 - CENTER_CROP))))
        return imagehash.phash(crop, hash_size=16).hash.flatten()


def collect(root):
    image_ids = sorted(p.stem for p in (root / "images").glob("*.jpg"))
    mask_ids = sorted(p.stem for p in (root / "masks").glob("*.jpg"))
    if image_ids != mask_ids:
        only_img = sorted(set(image_ids) - set(mask_ids))[:5]
        only_mask = sorted(set(mask_ids) - set(image_ids))[:5]
        raise ValueError(f"Снимки и маски не совпадают. Без маски: {only_img}; без снимка: {only_mask}")

    records = []
    for i, sample_id in enumerate(image_ids, 1):
        img_path = root / "images" / f"{sample_id}.jpg"
        with Image.open(img_path) as img:
            size = img.size
        with Image.open(root / "masks" / f"{sample_id}.jpg") as m:
            if m.size != size:
                raise ValueError(f"У {sample_id} размер маски {m.size} не совпадает со снимком {size}")
        mask = load_mask(root / "masks" / f"{sample_id}.jpg")
        records.append(
            {"id": sample_id, "width": size[0], "height": size[1], "area": float(mask.mean()), "hash": center_phash(img_path)}
        )
        if i % 200 == 0:
            print(f"[prepare] обработано {i}/{len(image_ids)}")
    return records


def find_duplicate_groups(records, threshold):
    """Пары с расстоянием Хэмминга <= threshold объединяются в группы (union-find)."""
    hashes = np.stack([r["hash"] for r in records])
    dist = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i, j in zip(*np.where(np.triu(dist <= threshold, k=1))):
        pairs.append((records[i]["id"], records[j]["id"], int(dist[i, j])))
        parent[find(i)] = find(j)
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values()), pairs


def size_bin(area):
    for k, (lo, hi) in enumerate(SIZE_BINS):
        if lo <= area < hi:
            return k
    return len(SIZE_BINS) - 1


def make_split(records, groups, seed, val_frac, test_frac):
    """Стратифицированное по размеру полипа разбиение групп."""
    rng = np.random.default_rng(seed)
    assignment = {}
    by_bin = {}
    for g in groups:
        by_bin.setdefault(size_bin(np.mean([records[i]["area"] for i in g])), []).append(g)
    for b in sorted(by_bin):
        bin_groups = by_bin[b]
        order = rng.permutation(len(bin_groups))
        n_items = sum(len(g) for g in bin_groups)
        targets = {"test": round(n_items * test_frac), "val": round(n_items * val_frac)}
        filled = {"test": 0, "val": 0}
        for idx in order:
            g = bin_groups[idx]
            if filled["test"] < targets["test"]:
                split = "test"
            elif filled["val"] < targets["val"]:
                split = "val"
            else:
                split = "train"
            if split != "train":
                filled[split] += len(g)
            for i in g:
                assignment[records[i]["id"]] = split
    return {s: sorted(i for i, sp in assignment.items() if sp == s) for s in SPLITS}


def verify_splits(splits_dir, records, groups):
    all_ids = {r["id"] for r in records}
    splits = {s: (Path(splits_dir) / f"{s}.txt").read_text(encoding="utf-8").split() for s in SPLITS}
    errors = []
    for s, ids in splits.items():
        missing = set(ids) - all_ids
        if missing:
            errors.append(f"{s}: {len(missing)} id нет в данных, например {sorted(missing)[0]}")
        if len(set(ids)) != len(ids):
            errors.append(f"{s}: есть повторяющиеся id")
    for a in SPLITS:
        for b in SPLITS:
            if a < b and set(splits[a]) & set(splits[b]):
                errors.append(f"выборки {a} и {b} пересекаются")
    covered = set().union(*splits.values())
    if covered != all_ids:
        errors.append(f"{len(all_ids - covered)} снимков не попали ни в одну выборку")
    where = {i: s for s, ids in splits.items() for i in ids}
    for g in groups:
        if len({where.get(records[i]["id"]) for i in g}) > 1:
            errors.append(f"группа дубликатов разорвана: {[records[i]['id'] for i in g]}")
    return splits, errors


def write_stats(records, splits, pairs, splits_dir, figures_dir):
    area = {r["id"]: r["area"] for r in records}
    sizes = Counter((r["width"], r["height"]) for r in records)
    stats = {
        "n_images": len(records),
        "image_width_range": [min(r["width"] for r in records), max(r["width"] for r in records)],
        "image_height_range": [min(r["height"] for r in records), max(r["height"] for r in records)],
        "n_unique_resolutions": len(sizes),
        "top_resolutions": [{"width": w, "height": h, "count": c} for (w, h), c in sizes.most_common(5)],
        "near_duplicate_pairs": len(pairs),
        "polyp_area_percent": {},
        "size_groups": {},
    }
    for s in ("all", *SPLITS):
        values = np.array([r["area"] for r in records] if s == "all" else [area[i] for i in splits[s]])
        stats["polyp_area_percent"][s] = {
            "n": int(len(values)),
            "mean": round(100 * float(values.mean()), 2),
            "p5": round(100 * float(np.percentile(values, 5)), 2),
            "median": round(100 * float(np.median(values)), 2),
            "p95": round(100 * float(np.percentile(values, 95)), 2),
            "min": round(100 * float(values.min()), 2),
            "max": round(100 * float(values.max()), 2),
        }
        stats["size_groups"][s] = {
            name: int(sum(lo <= v < hi for v in values))
            for name, (lo, hi) in zip(("small_lt_5pct", "medium_5_20pct", "large_gt_20pct"), SIZE_BINS)
        }
    (Path(splits_dir) / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # Кумулятивное распределение: на выборках по 100 снимков читается лучше гистограммы
    for s in SPLITS:
        values = np.sort([100 * area[i] for i in splits[s]])
        axes[0].step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=f"{s} (n={len(values)})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("площадь полипа, % кадра (лог. шкала)")
    axes[0].set_ylabel("доля снимков с полипом не больше x")
    axes[0].set_title("Размер полипа по выборкам (ECDF)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].scatter([r["width"] for r in records], [r["height"] for r in records], s=10, alpha=0.5)
    axes[1].set_xlabel("ширина, px")
    axes[1].set_ylabel("высота, px")
    axes[1].set_title(f"Разрешение снимков ({len(sizes)} уникальных)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "data_overview.png", dpi=120)
    plt.close(fig)
    return stats


def main():
    p = argparse.ArgumentParser(description="Подготовка разбиения Kvasir-SEG")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--figures-dir", default="reports/figures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--hash-threshold", type=int, default=56)
    p.add_argument("--force", action="store_true", help="пересоздать разбиение, даже если оно уже есть")
    args = p.parse_args()

    root = find_dataset_root(args.raw_dir)
    print(f"[prepare] данные: {root}")
    records = collect(root)
    groups, pairs = find_duplicate_groups(records, args.hash_threshold)
    print(f"[prepare] снимков: {len(records)}, пар почти-дубликатов: {len(pairs)}, групп: {len(groups)}")

    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    exists = all((splits_dir / f"{s}.txt").exists() for s in SPLITS)

    if exists and not args.force:
        splits, errors = verify_splits(splits_dir, records, groups)
        if errors:
            print("[prepare] разбиение НЕ прошло проверку:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)
        print("[prepare] существующее разбиение проверено: " + ", ".join(f"{s}={len(splits[s])}" for s in SPLITS))
        return

    splits = make_split(records, groups, args.seed, args.val_frac, args.test_frac)
    for s in SPLITS:
        (splits_dir / f"{s}.txt").write_text("\n".join(splits[s]) + "\n", encoding="utf-8")
    with (splits_dir / "duplicates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id_a", "id_b", "phash16_center_distance", "split"])
        where = {i: s for s, ids in splits.items() for i in ids}
        for a, b, d in pairs:
            writer.writerow([a, b, d, where[a]])

    archive = Path(args.raw_dir) / ARCHIVE_NAME
    manifest = {
        "source_url": KVASIR_URL,
        "archive_sha256": file_sha256(archive) if archive.exists() else KVASIR_SHA256,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "stratified_by": "доля площади полипа: <5%, 5-20%, >20%",
        "hash": f"pHash 16x16 по центральным {int(100 * (1 - 2 * CENTER_CROP))}% кадра",
        "hash_threshold": args.hash_threshold,
        "counts": {s: len(splits[s]) for s in SPLITS},
        "files_sha256": {s: file_sha256(splits_dir / f"{s}.txt") for s in SPLITS},
    }
    (splits_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = write_stats(records, splits, pairs, splits_dir, args.figures_dir)

    print("[prepare] разбиение: " + ", ".join(f"{s}={len(splits[s])}" for s in SPLITS))
    for s in SPLITS:
        a = stats["polyp_area_percent"][s]
        print(f"[prepare]   {s:5s} площадь полипа, %: медиана {a['median']}, p5 {a['p5']}, p95 {a['p95']}")
    print(f"[prepare] файлы: {splits_dir}")


if __name__ == "__main__":
    main()
