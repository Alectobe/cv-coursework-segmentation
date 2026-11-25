import pytest
import torch
import torch.nn.functional as F

from polypseg.losses import BinaryFocalLoss, DiceFocalLoss, SoftDiceLoss, build_loss
from polypseg.metrics import SegmentationMeter, confusion_counts, scores_from_counts


def make_target():
    target = torch.zeros(2, 1, 16, 16)
    target[0, :, 4:10, 4:10] = 1
    target[1, :, 0:3, 0:16] = 1
    return target


def test_perfect_prediction_scores_one():
    target = make_target()
    scores = scores_from_counts(*confusion_counts(target, target))
    for name in ("dice", "iou", "precision", "recall"):
        assert torch.allclose(scores[name], torch.ones(2, dtype=torch.float64))


def test_known_values():
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    pred[0, 0, 0, :2] = 1  # 2 пикселя
    target[0, 0, 0, 1:4] = 1  # 3 пикселя, пересечение 1
    s = scores_from_counts(*confusion_counts(pred, target))
    assert s["dice"].item() == pytest.approx(2 * 1 / (2 + 3))
    assert s["iou"].item() == pytest.approx(1 / 4)
    assert s["precision"].item() == pytest.approx(1 / 2)
    assert s["recall"].item() == pytest.approx(1 / 3)


def test_empty_cases():
    empty = torch.zeros(1, 1, 8, 8)
    full = torch.ones(1, 1, 8, 8)
    both_empty = scores_from_counts(*confusion_counts(empty, empty))
    assert both_empty["dice"].item() == 1 and both_empty["precision"].item() == 1
    missed = scores_from_counts(*confusion_counts(empty, full))
    assert missed["dice"].item() == 0 and missed["precision"].item() == 0 and missed["recall"].item() == 0


def test_meter_averages_per_image():
    target = make_target()
    logits = (target * 2 - 1) * 10
    logits[1] = -10  # второе изображение полностью пропущено
    meter = SegmentationMeter()
    meter.update(logits, target)
    assert meter.compute()["dice"] == pytest.approx(0.5)


def test_dice_loss_bounds():
    target = make_target()
    good = SoftDiceLoss()((target * 2 - 1) * 20, target)
    bad = SoftDiceLoss()((1 - target * 2) * 20, target)
    assert good.item() < 0.01
    assert bad.item() > 0.95


def test_focal_downweights_easy_pixels():
    target = make_target()
    logits = (target * 2 - 1) * 3  # уверенные правильные ответы
    bce = F.binary_cross_entropy_with_logits(logits, target)
    focal = BinaryFocalLoss(gamma=2.0)(logits, target)
    assert focal.item() < bce.item()
    assert BinaryFocalLoss(gamma=0.0)(logits, target).item() == pytest.approx(bce.item(), rel=1e-5)


def test_build_loss_and_backward():
    target = make_target()
    for name in ("bce", "dice", "focal", "dice_focal"):
        logits = torch.randn(2, 1, 16, 16, requires_grad=True)
        loss = build_loss({"name": name})(logits, target)
        loss.backward()
        assert torch.isfinite(loss) and logits.grad is not None
    assert isinstance(build_loss({"name": "dice_focal"}), DiceFocalLoss)
    with pytest.raises(ValueError):
        build_loss({"name": "unknown"})
