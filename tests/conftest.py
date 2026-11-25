import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from polypseg.config import load_config  # noqa: E402
from polypseg.models.factory import build_model  # noqa: E402

CONFIGS = ["baseline", "attn_gates", "dice_focal", "aspp", "attn_dice_focal"]


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def checkpoint_path(tmp_path_factory):
    """Чекпоинт в формате train.py со случайными весами (для тестов сервиса)."""
    cfg = load_config(ROOT / "configs" / "attn_gates.yaml")
    torch.manual_seed(0)
    model = build_model(cfg["model"], pretrained=False)
    path = tmp_path_factory.mktemp("weights") / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "state_dict": model.state_dict(),
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "experiment": "attn_gates",
            "epoch": 1,
            "val_metrics": {"dice": 0.5},
            "image_size": cfg["data"]["image_size"],
            "threshold": 0.5,
            "clearml": {"project": "test", "task_id": "test-task", "task_url": None},
        },
        path,
    )
    return path
