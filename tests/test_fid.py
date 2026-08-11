import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("scipy")

from metrics.fid import calculate_fid  # noqa: E402


def test_fid_identical_sets_is_zero():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(256, 64))
    assert calculate_fid(feats, feats) == pytest.approx(0.0, abs=1e-5)


def test_fid_shifted_means_scales_with_dimension():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(256, 32))
    b = rng.normal(loc=5.0, size=(256, 32))
    # same covariances, means differ by 5 in all 32 dims -> ~5^2 * 32
    assert calculate_fid(a, b) == pytest.approx(32 * 25, rel=0.15)


def test_fid_never_negative():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(128, 16))
    b = rng.normal(size=(128, 16))
    assert calculate_fid(a, b) >= 0.0
