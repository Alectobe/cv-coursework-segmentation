"""Сводная таблица экспериментов для отчёта.

    python scripts/compare_experiments.py

Читает artifacts/experiments/*/summary.json (+ history.csv) и, если есть,
artifacts/eval/<эксперимент>_val|test/metrics.json. Пишет:
- reports/experiments.md              - таблица сравнения с исходной моделью;
- reports/figures/val_dice_curves.png - val Dice по эпохам для всех вариантов;
- reports/figures/curves_<имя>.png    - loss и метрики train/val каждого варианта.
Запуски с переопределёнными параметрами (--epochs/--limit/--no-pretrained) в таблицу
не попадают: их условия отличаются от остальных.
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from polypseg.plots import read_history  # noqa: E402

BASELINE = "baseline"


def load_runs(exp_dir, eval_dir):
    runs = []
    for summary_path in sorted(Path(exp_dir).glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        overrides = {k: v for k, v in summary.get("overrides", {}).items() if k != "seed"}
        if overrides:
            print(f"[compare] пропускаю {summary['experiment']}: переопределены {overrides}")
            continue
        cfg = yaml.safe_load((summary_path.parent / "config.yaml").read_text(encoding="utf-8"))
        run = {"summary": summary, "config": cfg, "dir": summary_path.parent}
        for split in ("val", "test"):
            m = Path(eval_dir) / f"{summary['experiment']}_{split}" / "metrics.json"
            run[split] = json.loads(m.read_text(encoding="utf-8")) if m.exists() else None
        runs.append(run)
    return runs


def modification(cfg):
    parts = []
    if cfg["model"].get("attention_gates"):
        parts.append("Attention Gates")
    if cfg["model"].get("aspp"):
        parts.append("ASPP")
    if cfg["loss"]["name"] != "bce":
        parts.append({"dice_focal": "Dice+Focal loss"}.get(cfg["loss"]["name"], cfg["loss"]["name"]))
    return ", ".join(parts) or "- (исходная модель)"


def fmt(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments-dir", default="artifacts/experiments")
    p.add_argument("--eval-dir", default="artifacts/eval")
    p.add_argument("--out", default="reports/experiments.md")
    p.add_argument("--figure", default="reports/figures/val_dice_curves.png")
    args = p.parse_args()

    runs = load_runs(args.experiments_dir, args.eval_dir)
    if not runs:
        raise SystemExit("[compare] нет завершённых экспериментов в " + args.experiments_dir)
    base = next((r for r in runs if r["summary"]["experiment"] == BASELINE), None)
    base_dice = base["summary"]["best_val"]["dice"] if base else None

    lines = [
        "# Сравнение экспериментов",
        "",
        "Основная метрика - mDice на валидационной выборке (лучшая эпоха, вход 352x352).",
        "Колонки «orig» - та же val-выборка, но метрики посчитаны в исходном разрешении снимков (`polypseg.evaluate`).",
        "",
        "| Эксперимент | Модификация | Параметров, млн | Лучшая эпоха | val mDice | Δ к baseline | val mIoU "
        "| val Precision | val Recall | val mDice (orig) | test mDice | ClearML task |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        s, bv = r["summary"], r["summary"]["best_val"]
        delta = "-" if base_dice is None or s["experiment"] == BASELINE else f"{bv['dice'] - base_dice:+.4f}"
        task = s.get("clearml_task_url")
        task_cell = f"[{s['clearml_task_id'][:8]}]({task})" if task else "-"
        lines.append(
            f"| {s['experiment']} | {modification(r['config'])} | {s['trainable_params'] / 1e6:.2f} | {s['best_epoch']} "
            f"| {fmt(bv['dice'])} | {delta} | {fmt(bv['iou'])} | {fmt(bv['precision'])} | {fmt(bv['recall'])} "
            f"| {fmt(r['val']['dice']['mean'] if r['val'] else None)} "
            f"| {fmt(r['test']['dice']['mean'] if r['test'] else None)} | {task_cell} |"
        )
    lines += ["", "Одинаковые для всех запусков условия: см. `configs/baseline.yaml` (секции data и train)."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Кривые обучения каждого варианта копируются в reports/ (artifacts/ не хранится в git)
    figures_dir = Path(args.figure).parent
    figures_dir.mkdir(parents=True, exist_ok=True)
    for r in runs:
        curves = r["dir"] / "curves.png"
        if curves.exists():
            shutil.copyfile(curves, figures_dir / f"curves_{r['summary']['experiment']}.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in runs:
        h = read_history(r["dir"] / "history.csv")
        ax.plot(h["epoch"], h["val_dice"], label=r["summary"]["experiment"], linewidth=2 if r is base else 1.3)
    ax.set_xlabel("эпоха")
    ax.set_ylabel("val mDice")
    ax.set_title("Валидационный Dice по эпохам")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figure, dpi=120)
    plt.close(fig)
    print("\n".join(lines))
    print(f"\n[compare] таблица: {args.out}, график: {args.figure}")


if __name__ == "__main__":
    main()
