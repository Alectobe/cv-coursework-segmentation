"""Обучение модели по конфигу эксперимента.

Пример:
    python -m polypseg.train --config configs/baseline.yaml

Для каждого запуска в ClearML сохраняются: название и описание эксперимента,
полный конфиг, версия данных (SHA-256 архива и файлов разбиения), loss и метрики
на train/val по эпохам, кривые обучения и веса лучшей эпохи (OutputModel).
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from polypseg.config import file_sha256, load_config, save_config, set_seed
from polypseg.data.dataset import KvasirSegDataset, read_split
from polypseg.data.transforms import build_train_augment
from polypseg.losses import build_loss
from polypseg.metrics import SegmentationMeter
from polypseg.models.factory import build_model, count_parameters
from polypseg.plots import plot_history
from polypseg.tracking import Tracker

CHECKPOINT_FORMAT = 1


def parse_args():
    p = argparse.ArgumentParser(description="Обучение U-Net для сегментации полипов (Kvasir-SEG)")
    p.add_argument("--config", required=True, help="путь к YAML-конфигу эксперимента")
    p.add_argument("--seed", type=int, help="переопределить seed (к имени эксперимента добавится _seed<N>)")
    p.add_argument("--epochs", type=int, help="переопределить число эпох (для проверки кода)")
    p.add_argument("--limit", type=int, help="взять только первые N примеров train и val (для проверки кода)")
    p.add_argument("--device", help="cuda / cpu; по умолчанию cuda, если доступна")
    p.add_argument("--no-tracker", action="store_true", help="не логировать в ClearML (только для проверки кода)")
    p.add_argument("--no-pretrained", action="store_true", help="не скачивать веса ImageNet (только для проверки кода)")
    return p.parse_args()


def dataset_info(cfg):
    """Описание версии данных и разбиения для трекера."""
    splits_dir = Path(cfg["data"]["splits_dir"])
    manifest = json.loads((splits_dir / "split_manifest.json").read_text(encoding="utf-8"))
    info = {
        "name": "Kvasir-SEG",
        "source_url": manifest["source_url"],
        "archive_sha256": manifest["archive_sha256"],
        "split_seed": manifest["seed"],
        "dedup": f"pHash 16x16 по центру кадра, порог Хэмминга {manifest['hash_threshold']}",
    }
    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}.txt"
        info[f"{split}_size"] = manifest["counts"][split]
        info[f"{split}_sha256"] = file_sha256(path)
    return info


@torch.no_grad()
def run_validation(model, loader, criterion, device, use_amp, threshold):
    model.eval()
    meter = SegmentationMeter(threshold)
    loss_sum, count = 0.0, 0
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        loss_sum += loss.item() * images.size(0)
        count += images.size(0)
        meter.update(logits, masks)
    return loss_sum / count, meter.compute()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["train"]

    overrides = {}
    if args.seed is not None:
        tcfg["seed"] = args.seed
        overrides["seed"] = args.seed
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
        overrides["epochs"] = args.epochs
    if args.limit is not None:
        overrides["limit"] = args.limit
    if args.no_pretrained:
        cfg["model"]["encoder_weights"] = None
        overrides["no_pretrained"] = True

    exp_name = cfg["experiment"]["name"] + (f"_seed{args.seed}" if args.seed is not None else "")
    out_dir = Path(cfg["output"]["dir"]) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(tcfg["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(tcfg.get("amp", True)) and torch.device(device).type == "cuda"
    torch.backends.cudnn.benchmark = True
    print(f"[train] эксперимент={exp_name} device={device} amp={use_amp}")

    tags = list(cfg["experiment"]["tags"]) + (["override"] if overrides else [])
    tracker = Tracker(
        enabled=not args.no_tracker,
        project=cfg["tracker"]["project"],
        task_name=exp_name,
        tags=tags,
        description=cfg["experiment"]["description"],
    )
    clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    tracker.connect_configuration(clean_cfg, name="config")
    for section in ("data", "model", "loss", "train"):
        tracker.connect(dict(cfg[section]), name=section)
    ds_info = dataset_info(cfg)
    tracker.connect(dict(ds_info), name="dataset")
    if overrides:
        tracker.connect(dict(overrides), name="overrides")
    tracker.artifact("splits", Path(cfg["data"]["splits_dir"]))
    save_config(cfg, out_dir / "config.yaml")

    # Данные
    dcfg = cfg["data"]
    train_ids = read_split(dcfg["splits_dir"], "train")
    val_ids = read_split(dcfg["splits_dir"], "val")
    if args.limit:
        train_ids, val_ids = train_ids[: args.limit], val_ids[: args.limit]
    train_ds = KvasirSegDataset(dcfg["root"], train_ids, dcfg["image_size"], augment=build_train_augment())
    val_ds = KvasirSegDataset(dcfg["root"], val_ids, dcfg["image_size"])
    generator = torch.Generator().manual_seed(tcfg["seed"])
    loader_kwargs = {
        "num_workers": dcfg["num_workers"],
        "pin_memory": torch.device(device).type == "cuda",
        "persistent_workers": dcfg["num_workers"] > 0,
    }
    # drop_last: BatchNorm в ветке pooling у ASPP не работает на батче из одного примера
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, **loader_kwargs)
    print(f"[train] train={len(train_ds)} val={len(val_ds)}")

    # Модель, функция потерь, оптимизатор
    model = build_model(cfg["model"], pretrained=not args.no_pretrained).to(device)
    n_params = count_parameters(model)
    tracker.connect({"trainable_params": n_params}, name="model_stats")
    print(f"[train] обучаемых параметров: {n_params:,}")
    criterion = build_loss(cfg["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg["epochs"], eta_min=tcfg["lr"] * tcfg.get("min_lr_ratio", 0.01)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    threshold = tcfg["threshold"]

    history_path = out_dir / "history.csv"
    best_path = out_dir / "best.pt"
    fields = [
        "epoch", "lr", "train_loss", "val_loss",
        "train_dice", "val_dice", "train_iou", "val_iou",
        "val_precision", "val_recall", "epoch_time_s",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_dice, best_epoch, best_val = -1.0, 0, {}
    start = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        meter = SegmentationMeter(threshold)
        loss_sum, count = 0.0, 0
        for images, masks, _ in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * images.size(0)
            count += images.size(0)
            meter.update(logits.detach(), masks)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        train_loss = loss_sum / count
        train_metrics = meter.compute()
        val_loss, val_metrics = run_validation(model, val_loader, criterion, device, use_amp, threshold)
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_metrics["dice"],
            "val_dice": val_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "epoch_time_s": time.time() - t0,
        }
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        tracker.scalar("loss", "train", train_loss, epoch)
        tracker.scalar("loss", "val", val_loss, epoch)
        for metric in ("dice", "iou"):
            tracker.scalar(metric, "train", train_metrics[metric], epoch)
            tracker.scalar(metric, "val", val_metrics[metric], epoch)
        tracker.scalar("precision", "val", val_metrics["precision"], epoch)
        tracker.scalar("recall", "val", val_metrics["recall"], epoch)
        tracker.scalar("lr", "lr", lr, epoch)

        improved = val_metrics["dice"] > best_dice
        if improved:
            best_dice, best_epoch, best_val = val_metrics["dice"], epoch, dict(val_metrics, loss=val_loss)
            torch.save(
                {
                    "format_version": CHECKPOINT_FORMAT,
                    "state_dict": model.state_dict(),
                    "config": clean_cfg,
                    "experiment": exp_name,
                    "epoch": epoch,
                    "val_metrics": best_val,
                    "image_size": dcfg["image_size"],
                    "threshold": threshold,
                    "clearml": {
                        "project": cfg["tracker"]["project"],
                        "task_id": tracker.task_id,
                        "task_url": tracker.task_url,
                    },
                },
                best_path,
            )
        print(
            f"[train] epoch {epoch:3d}/{tcfg['epochs']} "
            f"loss {train_loss:.4f}/{val_loss:.4f} dice {train_metrics['dice']:.4f}/{val_metrics['dice']:.4f} "
            f"iou(val) {val_metrics['iou']:.4f} {row['epoch_time_s']:.1f}s" + (" *" if improved else "")
        )

    total_time = time.time() - start
    curves_path = out_dir / "curves.png"
    plot_history(history_path, curves_path, f"{exp_name}: обучение и валидация")
    tracker.image("curves", "train_val", curves_path, iteration=tcfg["epochs"])
    tracker.artifact("history", history_path)

    model_id = tracker.upload_model(best_path, name=f"{exp_name}_best", tags=tags, iteration=best_epoch)
    for name, value in best_val.items():
        tracker.single_value(f"best_val_{name}", value)

    summary = {
        "experiment": exp_name,
        "config": cfg["_source"],
        "best_epoch": best_epoch,
        "best_val": best_val,
        "epochs": tcfg["epochs"],
        "trainable_params": n_params,
        "train_time_min": round(total_time / 60, 2),
        "overrides": overrides,
        "clearml_task_id": tracker.task_id,
        "clearml_task_url": tracker.task_url,
        "clearml_model_id": model_id,
        "checkpoint": str(best_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.artifact("summary", summary)
    tracker.close()
    print(f"[train] готово: лучшая эпоха {best_epoch}, val Dice {best_dice:.4f}; результаты в {out_dir}")
    if model_id:
        print(f"[train] ClearML task: {summary['clearml_task_url']}  model id: {model_id}")


if __name__ == "__main__":
    main()
