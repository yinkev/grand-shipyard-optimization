from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


PROTOCOL = Path(__file__).resolve().parents[2] / "solver/final-release/c19_protocol.py"


def load_protocol(name: str = "c19_protocol_test"):
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, PROTOCOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def solution(block_id: int = 7) -> dict:
    return {
        "operations": {
            "3": [
                {
                    "type": "ENTRY",
                    "block_id": block_id,
                    "bay_id": 2,
                    "x": 11,
                    "y": 5,
                    "orient_idx": 1,
                }
            ],
            "9": [{"type": "EXIT", "block_id": block_id}],
        }
    }


def test_atomic_write_replaces_complete_json(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "payload.json"
    protocol.atomic_write_json(path, {"generation": 1})
    protocol.atomic_write_json(path, {"generation": 2})
    assert json.loads(path.read_text()) == {"generation": 2}
    assert not list(tmp_path.glob(".payload.json.tmp-*"))


def test_atomic_write_preserves_numeric_operation_chronology(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "payload.json"
    payload = {
        "solution": {
            "operations": {
                "2": [{"type": "ENTRY", "block_id": 0, "bay_id": 0}],
                "10": [{"type": "EXIT", "block_id": 0, "bay_id": 0}],
            }
        }
    }

    protocol.atomic_write_json(path, payload)

    loaded = json.loads(path.read_text())
    assert list(loaded["solution"]["operations"]) == ["2", "10"]


def test_publish_retains_only_strict_improvements_and_history(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "best.json"
    history = tmp_path / "history"

    first = protocol.publish_best(
        path, solution(), 100, "constructor", history_dir=history
    )
    assert first is not None and first["generation"] == 1
    assert protocol.publish_best(path, solution(), 100, "tie") is None
    assert protocol.publish_best(path, solution(), 101, "worse") is None
    second = protocol.publish_best(
        path,
        solution(),
        90,
        "repair",
        history_dir=history,
        metadata={"round": 2},
    )

    assert second is not None and second["generation"] == 2
    assert protocol.read_best(path)["objective"] == 90
    snapshots = protocol.list_history(history)
    assert [item["objective"] for item in snapshots] == [100, 90]
    assert snapshots[-1]["metadata"] == {"round": 2}


def test_concurrent_publishers_finish_at_minimum(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "best.json"
    objectives = [110, 80, 95, 70, 105, 60, 100]

    with ThreadPoolExecutor(max_workers=len(objectives)) as executor:
        list(
            executor.map(
                lambda objective: protocol.publish_best(
                    path, solution(), objective, f"worker-{objective}"
                ),
                objectives,
            )
        )

    assert protocol.read_best(path)["objective"] == min(objectives)


def test_corrupt_best_is_not_returned_and_can_be_recovered(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "best.json"
    path.write_text("{partial")
    assert protocol.read_best(path) is None
    assert protocol.publish_best(path, solution(), 42, "recovery") is not None
    assert protocol.read_best(path)["objective"] == 42


def test_validated_final_may_replace_equal_partial(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "best.json"
    assert protocol.publish_best(
        path, solution(), 42, "floor", metadata={"partial": True}
    )
    assert protocol.publish_best(path, solution(), 42, "floor") is None
    replaced = protocol.publish_best(
        path,
        solution(),
        42,
        "floor",
        metadata={"partial": False},
        replace_equal=True,
    )
    assert replaced is not None
    assert replaced["generation"] == 2
    assert replaced["metadata"] == {"partial": False}


@pytest.mark.parametrize("objective", [float("nan"), float("inf"), True, "1"])
def test_publish_rejects_non_finite_or_non_numeric_objective(tmp_path, objective):
    protocol = load_protocol()
    with pytest.raises((TypeError, ValueError)):
        protocol.publish_best(tmp_path / "best.json", solution(), objective, "x")


def test_parse_solution_records_extracts_entry_bay_and_exit():
    protocol = load_protocol()
    parsed = protocol.parse_solution_records(solution())
    assert parsed == {
        7: {
            "block_id": 7,
            "entry": 3,
            "exit": 9,
            "bay_id": 2,
            "x": 11,
            "y": 5,
            "orient_idx": 1,
        }
    }


@pytest.mark.parametrize(
    "bad_solution",
    [
        {"operations": {"1": [{"type": "EXIT", "block_id": 0}]}},
        {
            "operations": {
                "1": [
                    {
                        "type": "ENTRY",
                        "block_id": 0,
                        "bay_id": 0,
                        "x": 0,
                        "y": 0,
                        "orient_idx": 0,
                    },
                    {
                        "type": "ENTRY",
                        "block_id": 0,
                        "bay_id": 0,
                        "x": 0,
                        "y": 0,
                        "orient_idx": 0,
                    },
                ],
                "2": [{"type": "EXIT", "block_id": 0}],
            }
        },
        {"operations": {"1": [{"type": "MOVE", "block_id": 0}]}},
    ],
)
def test_parse_solution_records_rejects_incomplete_or_duplicate_data(
    bad_solution,
):
    protocol = load_protocol()
    with pytest.raises(ValueError):
        protocol.parse_solution_records(bad_solution)


def test_set_affinity_normalizes_and_calls_platform_api(monkeypatch):
    protocol = load_protocol()
    calls = []
    monkeypatch.setattr(
        protocol.os,
        "sched_setaffinity",
        lambda pid, cpus: calls.append((pid, cpus)),
        raising=False,
    )

    assert protocol.set_affinity("3,1,3") == (3, 1)
    assert calls == [(0, {1, 3})]
