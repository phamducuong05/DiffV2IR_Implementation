import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# model_utils pulls the full torch + ldm + blip stack; skip cleanly where
# those deps are not installed (e.g. the bare local dev env).
pytest.importorskip("torch")
pytest.importorskip("model_utils")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from ldm.modules.ema import LitEma  # noqa: E402
from model_utils import _ema_weights_present, apply_ema_for_inference  # noqa: E402


class _TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)


class _Holder(nn.Module):
    """Stand-in for LatentDiffusion: .model (EMA'd denoiser) + .model_ema."""

    def __init__(self):
        super().__init__()
        self.model = _TinyDenoiser()
        self.use_ema = True
        self.model_ema = LitEma(self.model)


def _flattened_ema_sd(holder, value=0.5):
    """Checkpoint state dict with LitEma-style flattened model_ema keys."""
    sd = {}
    for pname, p in holder.model.named_parameters():
        sname = pname.replace(".", "")
        sd[f"model_ema.{sname}"] = torch.full_like(p, value)
    return sd


def test_ema_weights_present():
    assert _ema_weights_present(
        {"model_ema.diffusion_modelinput_blocks00weight": torch.zeros(1)}) is True
    # dotted keys would NOT have loaded into LitEma's flattened buffers -> False
    assert _ema_weights_present(
        {"model_ema.diffusion_model.input_blocks.0.0.weight": torch.zeros(1)}) is False
    assert _ema_weights_present(
        {"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(1)}) is False
    assert _ema_weights_present({}) is False


def test_apply_ema_applies_and_drops():
    holder = _Holder()
    applied = apply_ema_for_inference(holder, _flattened_ema_sd(holder, value=0.5))
    assert applied is True
    assert holder.use_ema is False
    assert not hasattr(holder, "model_ema")
    for p in holder.model.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.5))


def test_apply_ema_without_checkpoint_ema_keeps_base():
    holder = _Holder()
    before = [p.clone() for p in holder.model.parameters()]
    applied = apply_ema_for_inference(holder, {})  # checkpoint has no model_ema keys
    assert applied is False
    assert holder.use_ema is False
    assert not hasattr(holder, "model_ema")
    for p, b in zip(holder.model.parameters(), before):
        assert torch.allclose(p, b)
