from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "solver/native-kernel"


def load_kernel():
    candidates = sorted(KERNEL_DIR.glob("_ogc2026_ckernel*.so"))
    if not candidates:
        pytest.skip("native kernel has not been built in place")
    spec = importlib.util.spec_from_file_location("_ogc2026_ckernel", candidates[0])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(forbidden, masks, H, W, h, w, ox, oy, x_lo, x_hi, y_lo, y_hi):
    rows = max(0, x_hi - x_lo + 1)
    cols = max(0, y_hi - y_lo + 1)
    result = np.zeros((rows, cols), dtype=bool)
    for i, x in enumerate(range(x_lo, x_hi + 1)):
        for j, y in enumerate(range(y_lo, y_hi + 1)):
            legal = 0 <= x + ox and x + ox + h <= H and 0 <= y + oy and y + oy + w <= W
            if not legal:
                continue
            ok = True
            for f, mask in zip(forbidden, masks, strict=True):
                region = f[x + ox:x + ox + h, y + oy:y + oy + w]
                if np.any(region & mask):
                    ok = False
                    break
            result[i, j] = ok
    return result


def test_compiled_kernel_matches_reference_on_random_cases():
    kernel = load_kernel()
    rng = np.random.default_rng(20260815)
    for _ in range(40):
        H, W, h, w, layers = 12, 17, 4, 5, 3
        forbidden = [rng.random((H, W)) < 0.18 for _ in range(layers)]
        masks = [rng.random((h, w)) < 0.45 for _ in range(layers)]
        actual = kernel.feasible_map_bitset(
            forbidden, masks, H, W, h, w, 0, 0, 0, H - h, 0, W - w
        )
        expected = reference(forbidden, masks, H, W, h, w, 0, 0, 0, H - h, 0, W - w)
        assert np.array_equal(actual, expected)
