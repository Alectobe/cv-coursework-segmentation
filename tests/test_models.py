import pytest
import torch

from polypseg.config import load_config
from polypseg.models.aspp import ASPP
from polypseg.models.attention_gate import AttentionGate
from polypseg.models.factory import build_model, count_parameters

from conftest import CONFIGS


@pytest.mark.parametrize("name", CONFIGS)
def test_forward_shape(root, name):
    cfg = load_config(root / "configs" / f"{name}.yaml")
    model = build_model(cfg["model"], pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 352, 352))
    assert out.shape == (2, 1, 352, 352)


def test_modifications_extend_baseline(root):
    """Attention Gates только добавляют параметры: все веса исходного U-Net остаются на своих местах."""
    base = build_model(load_config(root / "configs" / "baseline.yaml")["model"], pretrained=False)
    attn = build_model(load_config(root / "configs" / "attn_gates.yaml")["model"], pretrained=False)
    aspp = build_model(load_config(root / "configs" / "aspp.yaml")["model"], pretrained=False)
    base_keys = set(base.state_dict())
    assert base_keys <= set(attn.state_dict())
    assert any(k.startswith("decoder.attention_gates.") for k in attn.state_dict())
    assert any(k.startswith("decoder.center.branches.") for k in aspp.state_dict())
    assert count_parameters(base) < count_parameters(attn) < count_parameters(aspp)


def test_attention_gate_output():
    gate = AttentionGate(skip_channels=64, gating_channels=128)
    x, g = torch.randn(2, 64, 44, 44), torch.randn(2, 128, 22, 22)
    out = gate(x, g)
    assert out.shape == x.shape
    alpha = gate.last_attention
    assert alpha.shape == (2, 1, 44, 44)
    assert alpha.min() >= 0 and alpha.max() <= 1


def test_attention_gate_odd_sizes():
    gate = AttentionGate(skip_channels=32, gating_channels=64)
    out = gate(torch.randn(1, 32, 45, 45), torch.randn(1, 64, 23, 23))
    assert out.shape == (1, 32, 45, 45)


def test_aspp_keeps_shape():
    block = ASPP(512, 512).eval()
    with torch.no_grad():
        out = block(torch.randn(2, 512, 11, 11))
    assert out.shape == (2, 512, 11, 11)
