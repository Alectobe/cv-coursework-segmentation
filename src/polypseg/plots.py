"""Графики для отчёта: кривые обучения и визуализация предсказаний."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from polypseg.visualize import error_map, overlay  # noqa: E402


def read_history(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def plot_history(history_csv, out_path, title):
    """Loss и Dice/IoU на обучении и валидации по эпохам."""
    h = read_history(history_csv)
    epochs = h["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(epochs, h["train_loss"], label="train")
    axes[0].plot(epochs, h["val_loss"], label="val")
    axes[0].set_title("Функция потерь")

    axes[1].plot(epochs, h["train_dice"], label="train")
    axes[1].plot(epochs, h["val_dice"], label="val")
    best = int(np.argmax(h["val_dice"]))
    axes[1].scatter([epochs[best]], [h["val_dice"][best]], color="red", zorder=3, label=f"best val = {h['val_dice'][best]:.4f}")
    axes[1].set_title("Dice")

    axes[2].plot(epochs, h["train_iou"], label="train")
    axes[2].plot(epochs, h["val_iou"], label="val")
    axes[2].set_title("IoU")

    for ax in axes:
        ax.set_xlabel("эпоха")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_examples(rows, out_path, title):
    """rows: список словарей {image, gt, pred, caption}. Колонки: снимок | GT | предсказание | ошибки."""
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.4 * n), squeeze=False)
    headers = ["снимок", "разметка (GT)", "предсказание", "ошибки: TP / FP / FN"]
    for r, row in enumerate(rows):
        panels = [
            row["image"],
            overlay(row["image"], row["gt"], color=(0, 200, 0)),
            overlay(row["image"], row["pred"], color=(255, 200, 0)),
            error_map(row["gt"], row["pred"]),
        ]
        for c, panel in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(headers[c])
        axes[r][0].set_ylabel(row["caption"], fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
