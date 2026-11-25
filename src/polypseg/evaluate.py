"""Оценка обученной модели на выборке (по умолчанию test) в исходном разрешении снимков.

Пример:
    python -m polypseg.evaluate --checkpoint artifacts/experiments/baseline/best.pt --split test

Что сохраняется в --out-dir (по умолчанию artifacts/eval/<эксперимент>_<выборка>):
- metrics.json      - средние метрики, разбивка по размеру полипа, число провалов;
- per_image.csv     - метрики каждого снимка (для анализа ошибок);
- examples_best.png / examples_worst.png - лучшие и худшие предсказания;
- dice_vs_size.png  - зависимость Dice от площади полипа.
С трекером результаты логируются в отдельную задачу ClearML (тип testing),
дочернюю по отношению к задаче обучения этой модели.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from polypseg.data.dataset import read_split  # noqa: E402
from polypseg.data.transforms import load_mask, load_rgb  # noqa: E402
from polypseg.inference import load_checkpoint, predict_proba  # noqa: E402
from polypseg.metrics import confusion_counts, scores_from_counts  # noqa: E402
from polypseg.plots import plot_examples  # noqa: E402
from polypseg.tracking import Tracker  # noqa: E402

# Границы групп по доле площади кадра, занятой полипом
SIZE_BINS = (("small (<5%)", 0.0, 0.05), ("medium (5-20%)", 0.05, 0.20), ("large (>20%)", 0.20, 1.01))
FAILURE_DICE = 0.5


def parse_args():
    p = argparse.ArgumentParser(description="Оценка модели сегментации полипов")
    p.add_argument("--checkpoint", required=True, help="путь к best.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out-dir", help="куда сохранить результаты")
    p.add_argument("--num-examples", type=int, default=5, help="сколько лучших и худших примеров рисовать")
    p.add_argument("--device", help="cuda / cpu")
    p.add_argument("--no-tracker", action="store_true", help="не логировать в ClearML")
    return p.parse_args()


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "median": float(np.median(arr))}


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.checkpoint, device=device)
    cfg = meta["config"]
    exp_name = meta["experiment"]
    threshold = meta["threshold"]
    image_size = meta["image_size"]
    out_dir = Path(args.out_dir or f"artifacts/eval/{exp_name}_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    source_task_id = (meta.get("clearml") or {}).get("task_id")
    tracker = Tracker(
        enabled=not args.no_tracker,
        project=cfg["tracker"]["project"],
        task_name=f"{exp_name}_eval_{args.split}",
        task_type="testing",
        tags=list(cfg["experiment"].get("tags", [])) + [f"eval_{args.split}"],
        description=f"Оценка модели {exp_name} (эпоха {meta.get('epoch')}) на выборке {args.split}",
    )
    tracker.set_parent(source_task_id)
    tracker.connect(
        {
            "checkpoint": str(args.checkpoint),
            "experiment": exp_name,
            "source_task_id": source_task_id or "",
            "split": args.split,
            "threshold": threshold,
            "image_size": image_size,
        },
        name="evaluation",
    )

    root = Path(cfg["data"]["root"])
    ids = read_split(cfg["data"]["splits_dir"], args.split)
    rows, tp_total, fp_total, fn_total, times = [], 0.0, 0.0, 0.0, []
    for sample_id in ids:
        image = load_rgb(root / "images" / f"{sample_id}.jpg")
        gt = load_mask(root / "masks" / f"{sample_id}.jpg")
        t0 = time.perf_counter()
        prob = predict_proba(model, image, image_size, device)
        times.append(time.perf_counter() - t0)
        pred = (prob > threshold).astype(np.uint8)

        tp, fp, fn = confusion_counts(torch.from_numpy(pred)[None], torch.from_numpy(gt)[None])
        scores = {k: v.item() for k, v in scores_from_counts(tp, fp, fn).items()}
        tp_total += tp.item()
        fp_total += fp.item()
        fn_total += fn.item()
        rows.append(
            {
                "id": sample_id,
                "width": image.size[0],
                "height": image.size[1],
                **scores,
                "gt_area": float(gt.mean()),
                "pred_area": float(pred.mean()),
                "mean_conf_in_pred": float(prob[pred == 1].mean()) if pred.any() else 0.0,
            }
        )

    # Сводные метрики
    metrics = {name: summarize([r[name] for r in rows]) for name in ("dice", "iou", "precision", "recall")}
    metrics["pixel_dice_global"] = 2 * tp_total / max(2 * tp_total + fp_total + fn_total, 1)
    metrics["n_images"] = len(rows)
    metrics["n_failures_dice_lt_0.5"] = sum(r["dice"] < FAILURE_DICE for r in rows)
    metrics["n_empty_predictions"] = sum(r["pred_area"] == 0 for r in rows)
    metrics["by_size"] = {}
    for name, lo, hi in SIZE_BINS:
        group = [r for r in rows if lo <= r["gt_area"] < hi]
        metrics["by_size"][name] = {
            "n": len(group),
            "dice_mean": float(np.mean([r["dice"] for r in group])) if group else None,
            "iou_mean": float(np.mean([r["iou"] for r in group])) if group else None,
        }
    metrics["inference_ms_per_image"] = 1000 * float(np.mean(times))
    metrics["device"] = device
    metrics["experiment"] = exp_name
    metrics["split"] = args.split
    metrics["checkpoint_epoch"] = meta.get("epoch")
    metrics["source_task_id"] = source_task_id

    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Визуализация: лучшие и худшие примеры
    ranked = sorted(rows, key=lambda r: r["dice"])
    k = min(args.num_examples, len(ranked))
    for tag, selection in (("worst", ranked[:k]), ("best", ranked[::-1][:k])):
        panels = []
        for r in selection:
            image = load_rgb(root / "images" / f"{r['id']}.jpg")
            gt = load_mask(root / "masks" / f"{r['id']}.jpg")
            pred = (predict_proba(model, image, image_size, device) > threshold).astype(np.uint8)
            caption = f"{r['id'][:12]}\nDice {r['dice']:.3f}\nполип {100 * r['gt_area']:.1f}% кадра"
            panels.append({"image": np.asarray(image), "gt": gt, "pred": pred, "caption": caption})
        path = out_dir / f"examples_{tag}.png"
        plot_examples(panels, path, f"{exp_name}, {args.split}: {tag} {k}")
        tracker.image("examples", tag, path)

    # Dice в зависимости от размера полипа
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter([100 * r["gt_area"] for r in rows], [r["dice"] for r in rows], s=14, alpha=0.7)
    axes[0].axhline(FAILURE_DICE, color="red", linestyle="--", linewidth=1, label=f"Dice = {FAILURE_DICE}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("площадь полипа, % кадра (лог. шкала)")
    axes[0].set_ylabel("Dice")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].hist([r["dice"] for r in rows], bins=20, range=(0, 1))
    axes[1].set_xlabel("Dice")
    axes[1].set_ylabel("число снимков")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"{exp_name}, {args.split}: mDice {metrics['dice']['mean']:.4f}")
    fig.tight_layout()
    size_plot = out_dir / "dice_vs_size.png"
    fig.savefig(size_plot, dpi=120)
    plt.close(fig)
    tracker.image("analysis", "dice_vs_size", size_plot)

    for name in ("dice", "iou", "precision", "recall"):
        tracker.single_value(f"{args.split}_{name}", metrics[name]["mean"])
    tracker.single_value(f"{args.split}_pixel_dice_global", metrics["pixel_dice_global"])
    tracker.single_value(f"{args.split}_failures", metrics["n_failures_dice_lt_0.5"])
    tracker.artifact("metrics", out_dir / "metrics.json")
    tracker.artifact("per_image", out_dir / "per_image.csv")
    tracker.close()

    print(f"[eval] {exp_name} / {args.split}: {len(rows)} снимков")
    for name in ("dice", "iou", "precision", "recall"):
        m = metrics[name]
        print(f"[eval]   {name:9s} mean {m['mean']:.4f}  std {m['std']:.4f}  median {m['median']:.4f}")
    print(f"[eval]   провалов (Dice < {FAILURE_DICE}): {metrics['n_failures_dice_lt_0.5']}")
    for name, value in metrics["by_size"].items():
        dice = "-" if value["dice_mean"] is None else f"{value['dice_mean']:.4f}"
        print(f"[eval]   {name:15s} n={value['n']:3d} Dice {dice}")
    print(f"[eval] результаты: {out_dir}")


if __name__ == "__main__":
    main()
