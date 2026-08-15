from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKERS_PATH = ROOT / "solver/final-release/c19_workers.py"


def load_workers(name):
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, WORKERS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_problem(tmp_path):
    path = tmp_path / "problem.json"
    path.write_text(json.dumps({"blocks": [{}, {}]}), encoding="utf-8")
    return str(path)


def solution(tag):
    return {
        "test_tag": tag,
        "operations": {
            "0": [
                {"type": "ENTRY", "block_id": 0, "bay_id": 0,
                 "x": 0, "y": 0, "orient_idx": 0},
                {"type": "ENTRY", "block_id": 1, "bay_id": 0,
                 "x": 1, "y": 0, "orient_idx": 0},
            ],
            "1": [
                {"type": "EXIT", "block_id": 0, "bay_id": 0},
                {"type": "EXIT", "block_id": 1, "bay_id": 0},
            ],
        },
    }


class FakeProtocol:
    def __init__(self):
        self.rows = []
        self.best = None
        self.affinities = []
        self.events = {}
        self.json_values = {}
        self.history_args = []

    def set_affinity(self, cpus):
        pinned = tuple(cpus)
        self.affinities.append(pinned)
        return pinned

    def publish_best(
            self, _path, candidate, objective, source, *,
            history_dir=None, metadata=None):
        self.history_args.append(history_dir)
        if self.best is not None and float(objective) >= self.best["objective"]:
            return None
        row = {
            "generation": len(self.rows) + 1,
            "objective": float(objective),
            "source": source,
            "solution": candidate,
            "metadata": dict(metadata or {}),
        }
        self.best = row
        self.rows.append(row)
        event = self.events.get(float(objective))
        if event is not None:
            event.set()
        return row

    def read_best(self, _path):
        return self.best

    def atomic_write_json(self, path, payload):
        self.json_values[str(path)] = json.loads(json.dumps(payload))

    def read_json(self, path, default=None):
        return self.json_values.get(str(path), default)

    def parse_solution_records(self, _solution):
        return {
            0: {"entry": 0, "exit": 1},
            1: {"entry": 0, "exit": 1},
        }


class FakeConstructorCore:
    def __init__(self, protocol):
        self._INCUMBENT = {
            "solution": None, "objective": None, "obj1": None}
        self.protocol = protocol
        self.received_budget = None
        self.capped = False

    def _cap_thread_pools(self):
        self.capped = True

    def _validate(self, _prob, candidate):
        tag = candidate["test_tag"]
        objective = {"first": 10, "second": 7, "invalid": 2}[tag]
        return {
            "feasible": tag != "invalid",
            "objective": objective,
            "obj1": objective,
        }

    def _scored_full_armored_algorithm(self, _problem, timelimit):
        self.received_budget = timelimit
        first = solution("first")
        self._INCUMBENT.update(solution=first, objective=10, obj1=10)
        assert self.protocol.events[10.0].wait(1.0)
        invalid = solution("invalid")
        self._INCUMBENT.update(solution=invalid, objective=2, obj1=2)
        time.sleep(0.03)
        second = solution("second")
        self._INCUMBENT.update(solution=second, objective=7, obj1=7)
        assert self.protocol.events[7.0].wait(1.0)
        return second


def test_constructor_streams_only_valid_improvements(tmp_path):
    workers = load_workers("c19_workers_constructor")
    protocol = FakeProtocol()
    protocol.events = {10.0: threading.Event(), 7.0: threading.Event()}
    core = FakeConstructorCore(protocol)
    deadline = time.monotonic() + 1.5
    report = workers.run_constructor(
        write_problem(tmp_path), str(tmp_path / "run"), deadline, (7,),
        core_module=core, protocol_module=protocol, poll_interval=0.002)

    assert core.capped
    assert core.received_budget > 1.0
    assert protocol.affinities == [(7,)]
    assert [row["objective"] for row in protocol.rows] == [10.0, 7.0]
    assert all(row["source"] == "constructor" for row in protocol.rows)
    assert report["error"] is None


class FakeSubmission7Core(FakeConstructorCore):
    def _submission7_armored_algorithm(self, _problem, timelimit):
        self.received_budget = timelimit
        first = solution("first")
        self._INCUMBENT.update(solution=first, objective=10, obj1=10)
        assert self.protocol.events[10.0].wait(1.0)
        second = solution("second")
        self._INCUMBENT.update(solution=second, objective=7, obj1=7)
        assert self.protocol.events[7.0].wait(1.0)
        return second


def test_submission7_streams_current_instance_improvements(tmp_path):
    workers = load_workers("c19_workers_submission7")
    protocol = FakeProtocol()
    protocol.events = {10.0: threading.Event(), 7.0: threading.Event()}
    core = FakeSubmission7Core(protocol)
    report = workers.run_submission7(
        write_problem(tmp_path), str(tmp_path / "run"),
        time.monotonic() + 1.5, (8,),
        core_module=core, protocol_module=protocol, poll_interval=0.002)

    assert core.received_budget > 1.0
    assert protocol.affinities == [(8,)]
    assert [row["objective"] for row in protocol.rows] == [10.0, 7.0]
    assert all(row["source"] == "submission7" for row in protocol.rows)
    assert report["error"] is None


class FakeFloorCore(FakeConstructorCore):
    def __init__(self, protocol):
        super().__init__(protocol)
        self.validate_calls = 0

    def _validate(self, prob, candidate):
        self.validate_calls += 1
        return super()._validate(prob, candidate)

    def algorithm(self, _problem, timelimit):
        self.received_budget = timelimit
        first = solution("first")
        self._INCUMBENT.update(solution=first, objective=10, obj1=10)
        assert self.protocol.events[10.0].wait(1.0)
        second = solution("second")
        self._INCUMBENT.update(solution=second, objective=7, obj1=7)
        assert self.protocol.events[7.0].wait(1.0)
        return second


def test_floor_pins_two_cpus_and_uses_full_remaining_wall(tmp_path):
    workers = load_workers("c19_workers_floor")
    protocol = FakeProtocol()
    protocol.events = {10.0: threading.Event(), 7.0: threading.Event()}
    floor = FakeFloorCore(protocol)
    report = workers.run_floor(
        write_problem(tmp_path), str(tmp_path / "run"),
        time.monotonic() + 1.5, (10, 11),
        floor_module=floor, protocol_module=protocol)

    assert floor.received_budget > 1.0
    assert protocol.affinities == [(10, 11)]
    assert [row["objective"] for row in protocol.rows] == [10.0, 7.0]
    assert all(row["source"] == "floor" for row in protocol.rows)
    assert report["cpus"] == [10, 11]
    assert floor.validate_calls == 1
    assert report["error"] is None


class FakeCalendarCore:
    def __init__(self):
        self.capped = False

    def _cap_thread_pools(self):
        self.capped = True

    def _validate(self, _prob, candidate):
        tag = candidate["test_tag"]
        return {
            "feasible": tag != "invalid",
            "objective": {"invalid": 1, "calendar": 9}[tag],
            "obj1": 3,
        }

    def _cpsat_run_arm_inline(
            self, _problem, _budget, _started, retain=None, publish=None):
        publish([2, 4])
        retain({
            "solution": solution("invalid"), "objective": 1,
            "round": 1, "reason": "bad", "cpsat_status": "FEASIBLE"})
        retained = {
            "solution": solution("calendar"), "objective": 9,
            "round": 2, "reason": "arm-ok-r2",
            "cpsat_status": "FEASIBLE",
        }
        retain(retained)
        return retained


def test_calendar_publishes_valid_pack_with_master_entries(tmp_path):
    workers = load_workers("c19_workers_calendar")
    core = FakeCalendarCore()
    protocol = FakeProtocol()
    report = workers.run_calendar(
        write_problem(tmp_path), str(tmp_path / "run"),
        time.monotonic() + 1.0, (3,),
        core_module=core, protocol_module=protocol)

    assert core.capped
    assert len(protocol.rows) == 1
    row = protocol.rows[0]
    assert row["objective"] == 9.0
    assert row["source"] == "calendar"
    assert row["metadata"]["master_entries"] == [2, 4]
    assert row["metadata"]["calendar_round"] == 2
    channel = protocol.json_values[
        str(tmp_path / "run" / "calendar_latest.json")]
    assert channel["objective"] == 9.0
    assert channel["master_entries"] == [2, 4]
    assert channel["token"]
    assert protocol.history_args == [None, None]
    assert not (tmp_path / "run" / "history").exists()
    assert report["error"] is None


def test_calendar_channel_updates_even_when_global_best_is_better(tmp_path):
    workers = load_workers("c19_workers_calendar_channel")
    core = FakeCalendarCore()
    protocol = FakeProtocol()
    protocol.best = {
        "generation": 1,
        "objective": 4.0,
        "source": "constructor",
        "solution": solution("calendar"),
        "metadata": {},
    }
    report = workers.run_calendar(
        write_problem(tmp_path), str(tmp_path / "run"),
        time.monotonic() + 1.0, (3,),
        core_module=core, protocol_module=protocol)

    assert report["published"] == 0
    assert protocol.best["objective"] == 4.0
    channel = protocol.json_values[
        str(tmp_path / "run" / "calendar_latest.json")]
    assert channel["objective"] == 9.0
    assert channel["master_entries"] == [2, 4]


class StepClock:
    def __init__(self, start=0.0, step=0.2):
        self.value = float(start)
        self.step = float(step)

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


class FakeRepairCore:
    _cp_model = object()

    class _Model:
        def __init__(self, _prob):
            self.n = 2

    def __init__(self):
        self.reset_routes = []
        self.capped = False

    def _cap_thread_pools(self):
        self.capped = True

    def _reset_kernel_telemetry(self, route):
        self.reset_routes.append(route)

    def _validate(self, _prob, candidate):
        objective = 8 if candidate["test_tag"] == "repaired" else 10
        return {"feasible": True, "objective": objective, "obj1": objective}


class FakeOffense:
    def __init__(self):
        self.replayed = []
        self.slice_deadlines = []

    def replay_solution(self, _model, candidate):
        self.replayed.append(candidate)

    def run_joint_repair(
            self, _prob, _model, master_entries, deadline, _cp_model,
            _validate, retain, **kwargs):
        self.slice_deadlines.append(deadline)
        payload = {
            "solution": solution("repaired"),
            "objective": 8,
            "generation": kwargs["generation_start"] + 1,
            "telemetry": {"master_entries": master_entries},
        }
        retain(payload)
        return payload


def test_repair_replays_latest_best_and_publishes_strict_improvement(tmp_path):
    workers = load_workers("c19_workers_repair")
    protocol = FakeProtocol()
    protocol.best = {
        "generation": 4,
        "objective": 10.0,
        "source": "calendar",
        "solution": solution("first"),
        "metadata": {"master_entries": [2, 4]},
    }
    core = FakeRepairCore()
    offense = FakeOffense()
    clock = StepClock(step=1.0)
    report = workers.run_repair(
        write_problem(tmp_path), str(tmp_path / "run"), 9.0, (5,),
        core_module=core, offense_module=offense,
        protocol_module=protocol, clock=clock)

    assert core.capped
    assert core.reset_routes
    assert offense.replayed
    assert protocol.rows[-1]["source"] == "repair"
    assert protocol.rows[-1]["objective"] == 8.0
    assert protocol.rows[-1]["metadata"]["seed_generation"] == 4
    assert protocol.rows[-1]["metadata"]["master_entries"] == [2, 4]
    assert report["published"] == 1


def test_repair_consumes_calendar_channel_once_then_global_best(tmp_path):
    workers = load_workers("c19_workers_repair_channel")
    protocol = FakeProtocol()
    protocol.best = {
        "generation": 6,
        "objective": 8.0,
        "source": "constructor",
        "solution": solution("repaired"),
        "metadata": {},
    }
    channel_path = tmp_path / "run" / "calendar_latest.json"
    protocol.json_values[str(channel_path)] = {
        "schema_version": 1,
        "token": "calendar-1",
        "objective": 10.0,
        "solution": solution("first"),
        "master_entries": [2, 4],
    }
    core = FakeRepairCore()

    class NoImprovementOffense(FakeOffense):
        def __init__(self):
            super().__init__()
            self.seed_objectives = []
            self.master_rows = []

        def run_joint_repair(
                self, _prob, _model, master_entries, _deadline, _cp_model,
                _validate, _retain, **kwargs):
            self.seed_objectives.append(kwargs["seed_objective"])
            self.master_rows.append(master_entries)
            return {"solution": None, "objective": kwargs["seed_objective"]}

    offense = NoImprovementOffense()
    report = workers.run_repair(
        write_problem(tmp_path), str(tmp_path / "run"), 14.0, (5,),
        core_module=core, offense_module=offense,
        protocol_module=protocol, clock=StepClock(step=1.0))

    replayed_tags = [row["test_tag"] for row in offense.replayed]
    assert replayed_tags[0] == "first"
    assert replayed_tags.count("first") == 1
    assert "repaired" in replayed_tags[1:]
    assert offense.seed_objectives[0] == 10.0
    assert offense.master_rows[0] == [2, 4]
    assert report["errors"] == []


def test_repair_can_skip_calendar_and_use_distinct_role_and_seed(tmp_path):
    workers = load_workers("c19_workers_repair_global")

    class GlobalOnlyProtocol(FakeProtocol):
        def read_json(self, _path, default=None):
            raise AssertionError("calendar channel must not be read")

    protocol = GlobalOnlyProtocol()
    protocol.best = {
        "generation": 9,
        "objective": 10.0,
        "source": "constructor",
        "solution": solution("first"),
        "metadata": {"master_entries": [2, 4]},
    }

    class RecordingOffense(FakeOffense):
        def __init__(self):
            super().__init__()
            self.calls = []

        def run_joint_repair(
                self, _prob, _model, _master_entries, _deadline, _cp_model,
                _validate, _retain, **kwargs):
            self.calls.append(kwargs)
            return {"solution": None, "objective": kwargs["seed_objective"]}

    offense = RecordingOffense()
    report = workers.run_repair(
        write_problem(tmp_path), str(tmp_path / "run"), 7.0, (5,),
        core_module=FakeRepairCore(), offense_module=offense,
        protocol_module=protocol, clock=StepClock(step=1.0),
        prefer_calendar=False, seed_offset=17, role="repair-global")

    assert offense.calls
    assert all(call["role"] == "repair-global" for call in offense.calls)
    assert offense.calls[0]["seed"] == 20260615 + 17 + 104729
    assert report["source"] == "repair-global"
    assert report["errors"] == []


def test_repair_slice_cap_adapts_at_ninety_seconds():
    workers = load_workers("c19_workers_repair_long")
    assert workers._repair_slice_seconds(89.999) == 8.0
    assert workers._repair_slice_seconds(90.0) == 20.0
    assert workers._repair_slice_seconds(600.0) == 20.0
