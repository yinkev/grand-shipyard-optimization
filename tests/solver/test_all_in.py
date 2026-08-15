from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2] / "solver/final-release"
MODULE_PATH = PACKAGE / "myalgorithm.py"


def load_module():
    name = "candidate23_c2pm_portfolio_module"
    sys.modules.pop(name, None)
    sys.path.insert(0, str(PACKAGE))
    try:
        spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(PACKAGE))


def test_high_pressure_uses_constructor_calendar_repair_and_c2pm():
    module = load_module()

    assert module._runtime_roles(
        (0, 1, 2, 3), 1_000_000_000, 60, 0.70, 250
    ) == [
        ("constructor", (0,)),
        ("calendar", (1,)),
        ("repair-calendar", (2,)),
        ("c2pm", (3,)),
    ]


def test_low_pressure_preserves_candidate19_topology():
    module = load_module()

    assert module._runtime_roles(
        (0, 1, 2, 3), 1_000_000_000, 60, 0.69, 250
    ) == [
        ("floor", (0, 1)),
        ("calendar", (2,)),
        ("repair", (3,)),
    ]


def test_small_high_pressure_preserves_candidate19_topology():
    module = load_module()

    assert module._runtime_roles(
        (0, 1, 2, 3), 1_000_000_000, 60, 1.20, 200
    ) == [
        ("floor", (0, 1)),
        ("calendar", (2,)),
        ("repair", (3,)),
    ]
