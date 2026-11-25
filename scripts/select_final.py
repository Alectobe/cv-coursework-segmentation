"""Выбор итоговой модели по val mDice и копирование её весов в artifacts/final/model.pt.

    python scripts/select_final.py                      # лучший по val mDice среди всех экспериментов
    python scripts/select_final.py --experiment aspp    # выбрать конкретный эксперимент

Выбор делается ТОЛЬКО по валидационной выборке: тестовая используется после выбора,
для итоговой оценки (python -m polypseg.evaluate --split test).
Рядом с весами сохраняется model_card.json: эксперимент, эпоха, метрики, ID в ClearML.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from polypseg.config import file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments-dir", default="artifacts/experiments")
    p.add_argument("--experiment", help="имя эксперимента; по умолчанию - лучший по val mDice")
    p.add_argument("--out", default="artifacts/final/model.pt")
    args = p.parse_args()

    candidates = []
    for path in sorted(Path(args.experiments_dir).glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        overrides = {k: v for k, v in summary.get("overrides", {}).items() if k != "seed"}
        if overrides:
            continue  # запуски для проверки кода (--epochs/--limit/--no-pretrained) не участвуют
        candidates.append(summary)
    if not candidates:
        sys.exit(f"[select] в {args.experiments_dir} нет завершённых экспериментов")

    if args.experiment:
        chosen = next((s for s in candidates if s["experiment"] == args.experiment), None)
        if chosen is None:
            sys.exit(f"[select] эксперимент {args.experiment} не найден среди {[s['experiment'] for s in candidates]}")
    else:
        chosen = max(candidates, key=lambda s: s["best_val"]["dice"])

    for s in sorted(candidates, key=lambda s: -s["best_val"]["dice"]):
        mark = "  <- выбран" if s is chosen else ""
        print(f"[select] {s['experiment']:20s} val mDice {s['best_val']['dice']:.4f}{mark}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(args.experiments_dir) / chosen["experiment"] / "best.pt", out)
    card = {
        "experiment": chosen["experiment"],
        "best_epoch": chosen["best_epoch"],
        "val_metrics": chosen["best_val"],
        "selection_rule": "максимальный mDice на val" if not args.experiment else "выбран вручную",
        "clearml_task_id": chosen.get("clearml_task_id"),
        "clearml_task_url": chosen.get("clearml_task_url"),
        "clearml_model_id": chosen.get("clearml_model_id"),
        "sha256": file_sha256(out),
    }
    (out.parent / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[select] веса: {out}\n[select] паспорт модели: {out.parent / 'model_card.json'}")
    print(json.dumps(card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
