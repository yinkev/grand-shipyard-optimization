"""Candidate 19 worker processes.

Each entry point owns one CPU, publishes only checker-valid complete
solutions, and treats the shared incumbent as an append-only improvement
stream.  Imports of the numerical engines are deliberately lazy so the
orchestrator can fork before loading their native runtimes.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _dependencies(
        core_module=None, offense_module=None, protocol_module=None, *,
        need_offense=True):
    if core_module is None:
        import candidate19_core as core_module
    if offense_module is None and need_offense:
        import frontier_offense as offense_module
    if protocol_module is None:
        import c19_protocol as protocol_module
    return core_module, offense_module, protocol_module


def _prepare(core, protocol, cpus) -> tuple[int, ...]:
    for name in _THREAD_ENV:
        os.environ[name] = "1"
    pinned = protocol.set_affinity(tuple(int(cpu) for cpu in cpus))
    cap = getattr(core, "_cap_thread_pools", None)
    if callable(cap):
        cap()
    return tuple(pinned)


def _load_problem(problem_path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(problem_path, "r", encoding="utf-8") as handle:
        problem = json.load(handle)
    if not isinstance(problem, dict):
        raise ValueError("problem payload must be a JSON object")
    return problem


def _paths(root_path: str | os.PathLike[str]) -> tuple[Path, None]:
    root = Path(root_path)
    root.mkdir(parents=True, exist_ok=True)
    return root / "best.json", None


def _clone_solution(solution) -> dict[str, Any] | None:
    if not isinstance(solution, dict):
        return None
    try:
        cloned = json.loads(json.dumps(solution, separators=(",", ":")))
    except (TypeError, ValueError):
        return None
    return cloned if isinstance(cloned, dict) else None


def _json_metadata(metadata) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(
            dict(metadata or {}), default=str, separators=(",", ":")))
    except (TypeError, ValueError):
        return {}


def _official_candidate(core, problem, solution):
    frozen = _clone_solution(solution)
    if frozen is None:
        return None
    try:
        report = core._validate(problem, frozen)
    except BaseException:
        return None
    if not report or not report.get("feasible"):
        return None
    objective = report.get("objective")
    if objective is None:
        return None
    try:
        objective = float(objective)
    except (TypeError, ValueError):
        return None
    return frozen, objective, report


def _publish_checked(
    core,
    protocol,
    problem,
    best_path,
    history_dir,
    source,
    solution,
    *,
    metadata=None,
    replace_equal=False,
):
    checked = _official_candidate(core, problem, solution)
    if checked is None:
        return None
    frozen, objective, report = checked
    detail = _json_metadata(metadata)
    for key in ("obj1", "obj2", "obj3"):
        if report.get(key) is not None:
            detail[key] = report[key]
    try:
        publish_options = {
            "history_dir": history_dir,
            "metadata": detail,
        }
        if replace_equal:
            publish_options["replace_equal"] = True
        return protocol.publish_best(
            best_path,
            frozen,
            objective,
            source,
            **publish_options,
        )
    except Exception:
        return None


def _run_incumbent_engine(
    problem_path,
    root_path,
    deadline,
    cpus,
    *,
    engine,
    protocol,
    source,
    runner,
    poll_interval,
    clock,
):
    pinned = _prepare(engine, protocol, cpus)
    problem = _load_problem(problem_path)
    best_path, history_dir = _paths(root_path)
    now = clock or time.monotonic
    stopped = threading.Event()
    observed = set()
    published = 0

    def publish_incumbent():
        nonlocal published
        incumbent = getattr(engine, "_INCUMBENT", {})
        solution = incumbent.get("solution")
        claimed = incumbent.get("objective")
        if solution is None or claimed is None:
            return
        try:
            marker = (float(claimed), id(solution))
        except (TypeError, ValueError):
            return
        if marker in observed:
            return
        observed.add(marker)
        row = _publish_checked(
            engine,
            protocol,
            problem,
            best_path,
            history_dir,
            source,
            solution,
            metadata={
                "claimed_objective": claimed,
                "worker_cpus": list(pinned),
            },
        )
        if row is not None:
            published += 1

    def monitor():
        while not stopped.is_set() and now() < float(deadline):
            publish_incumbent()
            stopped.wait(max(0.001, float(poll_interval)))
        publish_incumbent()

    watcher = threading.Thread(
        target=monitor, name=f"c19-{source}-incumbent", daemon=True)
    watcher.start()
    result = None
    error = None
    started = now()
    try:
        remaining = max(0.0, float(deadline) - started)
        if remaining > 0.0:
            result = runner(problem, remaining)
            _publish_checked(
                engine,
                protocol,
                problem,
                best_path,
                history_dir,
                source,
                result,
                metadata={
                    "returned_solution": True,
                    "worker_cpus": list(pinned),
                },
            )
    except BaseException as exc:
        error = f"{type(exc).__name__}:{str(exc)[:160]}"
    finally:
        publish_incumbent()
        stopped.set()
        watcher.join(timeout=max(0.05, 2.0 * float(poll_interval)))
        publish_incumbent()
    return {
        "source": source,
        "published": published,
        "returned_solution": isinstance(result, dict),
        "error": error,
        "wall_s": max(0.0, now() - started),
        "cpus": list(pinned),
    }


def run_constructor(
    problem_path: str,
    root_path: str,
    deadline: float,
    cpus: tuple[int, ...],
    *,
    core_module=None,
    protocol_module=None,
    poll_interval: float = 0.01,
    clock=None,
):
    """Run the full scored constructor/LNS arm and stream its incumbents."""
    core, _offense, protocol = _dependencies(
        core_module, None, protocol_module, need_offense=False)
    return _run_incumbent_engine(
        problem_path,
        root_path,
        deadline,
        cpus,
        engine=core,
        protocol=protocol,
        source="constructor",
        runner=lambda problem, remaining: core._scored_full_armored_algorithm(
            problem, timelimit=remaining),
        poll_interval=poll_interval,
        clock=clock,
    )


def run_submission7(
    problem_path: str,
    root_path: str,
    deadline: float,
    cpus: tuple[int, ...],
    *,
    core_module=None,
    protocol_module=None,
    poll_interval: float = 0.01,
    clock=None,
):
    """Run the hidden-proven Submission 7 arm and stream its incumbents."""
    core, _offense, protocol = _dependencies(
        core_module, None, protocol_module, need_offense=False)
    return _run_incumbent_engine(
        problem_path,
        root_path,
        deadline,
        cpus,
        engine=core,
        protocol=protocol,
        source="submission7",
        runner=lambda problem, remaining:
            core._submission7_armored_algorithm(
                problem, timelimit=remaining),
        poll_interval=poll_interval,
        clock=clock,
    )


def run_c2pm(
    problem_path: str,
    root_path: str,
    deadline: float,
    cpus: tuple[int, ...],
    *,
    c2pm_module=None,
    protocol_module=None,
):
    """Run Candidate 11's C2PM plus causal-repair lane on one core."""
    if c2pm_module is None:
        import candidate11_core as c2pm_module
    if protocol_module is None:
        import c19_protocol as protocol_module
    pinned = _prepare(c2pm_module, protocol_module, cpus)
    problem = _load_problem(problem_path)
    best_path, history_dir = _paths(root_path)
    started = time.monotonic()
    published = 0
    error = None

    def retain(payload):
        nonlocal published
        if not isinstance(payload, dict):
            return
        solution = payload.get("solution")
        if not isinstance(solution, dict):
            return
        row = _publish_checked(
            c2pm_module,
            protocol_module,
            problem,
            best_path,
            history_dir,
            "c2pm",
            solution,
            metadata={
                "c2pm_role": payload.get("role"),
                "c2pm_generation": payload.get("generation"),
                "c2pm_alpha": payload.get("alpha"),
                "c2pm_telemetry": payload.get("telemetry"),
                "worker_cpus": list(pinned),
            },
        )
        if row is not None:
            published += 1

    result = None
    try:
        remaining = max(0.1, float(deadline) - time.monotonic() - 0.10)
        layers = c2pm_module._cpsat_n_layers(
            problem.get("blocks") or ())
        role = "nominal" if layers >= 3 else "combined"
        if layers >= 3:
            os.environ["C11_SEED_BUDGET_S"] = str(
                min(40.0, max(0.1, remaining - 5.0)))
        result = c2pm_module._c11_run_role(
            problem,
            role,
            remaining,
            time.monotonic(),
            retain=retain,
            strong_cumulative=False,
        )
        retain(result)
    except BaseException as exc:
        error = f"{type(exc).__name__}:{str(exc)[:160]}"
    return {
        "source": "c2pm",
        "published": published,
        "returned_solution": (
            isinstance(result, dict)
            and isinstance(result.get("solution"), dict)
        ),
        "error": error,
        "wall_s": max(0.0, time.monotonic() - started),
        "cpus": list(pinned),
    }


def run_floor(
    problem_path: str,
    root_path: str,
    deadline: float,
    cpus: tuple[int, ...],
    *,
    floor_module=None,
    protocol_module=None,
    poll_interval: float = 0.05,
    clock=None,
):
    """Run exact S8 with a copy-only monitor on its dedicated two CPUs."""
    if floor_module is None:
        import submission8_floor as floor_module
    if protocol_module is None:
        import c19_protocol as protocol_module
    pinned = _prepare(floor_module, protocol_module, cpus)
    problem = _load_problem(problem_path)
    best_path, history_dir = _paths(root_path)
    now = clock or time.monotonic
    stopped = threading.Event()
    observed = set()
    published = 0

    def publish_snapshot():
        nonlocal published
        incumbent = dict(getattr(floor_module, "_INCUMBENT", {}))
        solution = incumbent.get("solution")
        claimed = incumbent.get("objective")
        if solution is None or claimed is None:
            return
        try:
            objective = float(claimed)
            marker = (objective, id(solution))
        except (TypeError, ValueError):
            return
        if marker in observed:
            return
        observed.add(marker)
        frozen = _clone_solution(solution)
        if frozen is None:
            return
        try:
            row = protocol_module.publish_best(
                best_path,
                frozen,
                objective,
                "floor",
                history_dir=history_dir,
                metadata={
                    "claimed_objective": claimed,
                    "copy_only_monitor": True,
                    "worker_cpus": list(pinned),
                },
            )
        except Exception:
            row = None
        if row is not None:
            published += 1

    def monitor():
        while not stopped.is_set() and now() < float(deadline):
            publish_snapshot()
            stopped.wait(max(0.001, float(poll_interval)))
        publish_snapshot()

    watcher = threading.Thread(
        target=monitor, name="c19-floor-incumbent", daemon=True)
    watcher.start()
    result = None
    error = None
    final_valid = False
    final_objective = None
    started = now()
    try:
        remaining = max(0.0, float(deadline) - started)
        if remaining > 0.0:
            result = floor_module.algorithm(problem, timelimit=remaining)
    except BaseException as exc:
        error = f"{type(exc).__name__}:{str(exc)[:160]}"
    finally:
        stopped.set()
        watcher.join(timeout=max(0.10, 2.0 * float(poll_interval)))
        publish_snapshot()
    if isinstance(result, dict):
        checked = _official_candidate(floor_module, problem, result)
        if checked is not None:
            frozen, final_objective, report = checked
            final_valid = True
            detail = {"returned_solution": True, "worker_cpus": list(pinned)}
            for key in ("obj1", "obj2", "obj3"):
                if report.get(key) is not None:
                    detail[key] = report[key]
            try:
                protocol_module.publish_best(
                    best_path,
                    frozen,
                    final_objective,
                    "floor",
                    history_dir=history_dir,
                    metadata=detail,
                    replace_equal=True,
                )
            except Exception:
                pass
    return {
        "source": "floor",
        "published": published,
        "returned_solution": isinstance(result, dict),
        "final_valid": final_valid,
        "final_objective": final_objective,
        "last_incumbent_objective": (
            getattr(floor_module, "_INCUMBENT", {}).get("objective")
        ),
        "error": error,
        "wall_s": max(0.0, now() - started),
        "cpus": list(pinned),
    }


def run_calendar(
    problem_path: str,
    root_path: str,
    deadline: float,
    cpus: tuple[int, ...],
    *,
    core_module=None,
    protocol_module=None,
    clock=None,
):
    """Run the calendar/decode arm and publish every retained valid pack."""
    core, _offense, protocol = _dependencies(
        core_module, None, protocol_module, need_offense=False)
    pinned = _prepare(core, protocol, cpus)
    problem = _load_problem(problem_path)
    best_path, history_dir = _paths(root_path)
    now = clock or time.monotonic
    started = now()
    master_entries = {"value": None}
    retained = 0
    published = 0
    channel_generation = 0
    error = None

    def capture_master(entries):
        if entries is None:
            return
        master_entries["value"] = [
            None if value is None else int(value) for value in entries]

    def retain(pack):
        nonlocal retained, published, channel_generation
        if not isinstance(pack, dict):
            return
        solution = pack.get("solution")
        if solution is None:
            return
        checked = _official_candidate(core, problem, solution)
        if checked is None:
            return
        frozen, objective, report = checked
        retained += 1
        entries = master_entries["value"]
        if entries is None:
            entries = _record_master_entries(
                protocol, {"solution": frozen, "metadata": {}},
                len(problem.get("blocks") or ()))
        metadata = {
            "master_entries": entries,
            "calendar_round": pack.get("round"),
            "calendar_reason": pack.get("reason"),
            "cpsat_status": pack.get("cpsat_status"),
            "worker_cpus": list(pinned),
        }
        for key in ("obj1", "obj2", "obj3"):
            if report.get(key) is not None:
                metadata[key] = report[key]
        channel_generation += 1
        try:
            protocol.atomic_write_json(
                best_path.parent / "calendar_latest.json",
                {
                    "schema_version": 1,
                    "token": (
                        f"{os.getpid()}:{time.time_ns()}:"
                        f"{channel_generation}"
                    ),
                    "objective": objective,
                    "solution": frozen,
                    "master_entries": entries,
                },
            )
        except Exception:
            pass
        try:
            row = protocol.publish_best(
                best_path,
                frozen,
                objective,
                "calendar",
                history_dir=history_dir,
                metadata=_json_metadata(metadata),
            )
        except Exception:
            row = None
        if row is not None:
            published += 1

    result = None
    try:
        remaining = max(0.0, float(deadline) - started)
        if remaining > 0.0:
            result = core._cpsat_run_arm_inline(
                problem,
                remaining,
                started,
                retain=retain,
                publish=capture_master,
            )
            if isinstance(result, dict) and result.get("solution") is not None:
                retain(result)
    except BaseException as exc:
        error = f"{type(exc).__name__}:{str(exc)[:160]}"
    return {
        "source": "calendar",
        "retained": retained,
        "published": published,
        "error": error,
        "wall_s": max(0.0, now() - started),
        "cpus": list(pinned),
    }


def _record_master_entries(protocol, record, n_blocks):
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        raw = metadata.get("master_entries")
        if isinstance(raw, list) and len(raw) == int(n_blocks):
            return [None if value is None else int(value) for value in raw]
    try:
        rows = protocol.parse_solution_records(record["solution"])
        return [int(rows[bid]["entry"]) for bid in range(int(n_blocks))]
    except Exception:
        return None


def _repair_slice_seconds(remaining: float) -> float:
    return 20.0 if float(remaining) >= 90.0 else 8.0


def run_repair(
    problem_path: str,
    root_path: str,
    deadline: float,
    cpus: tuple[int, ...],
    *,
    core_module=None,
    offense_module=None,
    protocol_module=None,
    clock=None,
    idle_sleep: float = 0.02,
    prefer_calendar: bool = True,
    seed_offset: int = 0,
    role: str = "repair",
    operator_phase: int = 0,
):
    """Continuously repair the freshest global incumbent in bounded slices."""
    core, offense, protocol = _dependencies(
        core_module, offense_module, protocol_module)
    pinned = _prepare(core, protocol, cpus)
    problem = _load_problem(problem_path)
    best_path, history_dir = _paths(root_path)
    now = clock or time.monotonic
    started = now()
    slices = 0
    published = 0
    errors = []
    calendar_path = Path(root_path) / "calendar_latest.json"
    last_calendar_token = None

    while True:
        remaining = float(deadline) - now()
        if remaining < 0.45:
            break
        record = None
        seed_channel = "global-best"
        if prefer_calendar:
            try:
                calendar = protocol.read_json(calendar_path)
            except Exception:
                calendar = None
            if isinstance(calendar, dict):
                token = calendar.get("token")
                if token is not None and token != last_calendar_token:
                    last_calendar_token = token
                    if (isinstance(calendar.get("solution"), dict)
                            and calendar.get("objective") is not None):
                        record = {
                            "generation": 0,
                            "objective": calendar["objective"],
                            "source": "calendar-channel",
                            "solution": calendar["solution"],
                            "metadata": {
                                "master_entries": calendar.get(
                                    "master_entries"),
                            },
                        }
                        seed_channel = "calendar-channel"
        if record is None:
            if (prefer_calendar and role == "repair-calendar"
                    and last_calendar_token is None):
                time.sleep(min(
                    max(0.001, idle_sleep), max(0.0, remaining)))
                continue
            record = protocol.read_best(best_path)
        if not record or record.get("solution") is None:
            time.sleep(min(max(0.001, idle_sleep), max(0.0, remaining)))
            continue
        try:
            seed_objective = float(record["objective"])
            generation = int(record.get("generation") or 0)
            solution = _clone_solution(record["solution"])
            if solution is None:
                raise ValueError("non-dict incumbent solution")

            reset = getattr(core, "_reset_kernel_telemetry", None)
            if callable(reset):
                reset("candidate19/repair")
            model = core._Model(problem)
            offense.replay_solution(model, solution)
            master_entries = _record_master_entries(
                protocol, record, model.n)
            slice_cap = _repair_slice_seconds(remaining)
            slice_deadline = min(
                float(deadline) - 0.10, now() + slice_cap)
            if slice_deadline <= now() + 0.20:
                break
            slices += 1

            def validate_fn(prob, candidate):
                try:
                    return core._validate(prob, candidate)
                except BaseException:
                    return None

            def retain(payload):
                nonlocal published
                if not isinstance(payload, dict):
                    return
                candidate = payload.get("solution")
                claimed = payload.get("objective")
                if candidate is None or claimed is None:
                    return
                checked = _official_candidate(core, problem, candidate)
                if checked is None:
                    return
                frozen, objective, report = checked
                if objective >= seed_objective - 1e-9:
                    return
                metadata = {
                    "seed_generation": generation,
                    "seed_objective": seed_objective,
                    "seed_channel": seed_channel,
                    "master_entries": master_entries,
                    "repair_generation": payload.get("generation"),
                    "repair_telemetry": payload.get("telemetry"),
                    "worker_cpus": list(pinned),
                }
                for key in ("obj1", "obj2", "obj3"):
                    if report.get(key) is not None:
                        metadata[key] = report[key]
                try:
                    row = protocol.publish_best(
                        best_path,
                        frozen,
                        objective,
                        role,
                        history_dir=history_dir,
                        metadata=_json_metadata(metadata),
                    )
                except Exception:
                    row = None
                if row is not None:
                    published += 1

            outcome = offense.run_joint_repair(
                problem,
                model,
                master_entries,
                slice_deadline,
                core._cp_model,
                validate_fn,
                retain,
                role=role,
                generation_start=generation,
                seed_objective=seed_objective,
                seed=20260615 + seed_offset + 104729 * slices,
                operator_phase=operator_phase,
            )
            if (isinstance(outcome, dict)
                    and outcome.get("solution") is not None
                    and outcome.get("objective") is not None
                    and float(outcome["objective"]) < seed_objective - 1e-9):
                retain(outcome)
        except BaseException as exc:
            errors.append(f"{type(exc).__name__}:{str(exc)[:160]}")
            if len(errors) > 16:
                del errors[:-16]

    return {
        "source": role,
        "slices": slices,
        "published": published,
        "errors": errors,
        "wall_s": max(0.0, now() - started),
        "cpus": list(pinned),
    }


# Explicit names for callers that prefer role-oriented entry points.
constructor_worker = run_constructor
submission7_worker = run_submission7
c2pm_worker = run_c2pm
floor_worker = run_floor
calendar_worker = run_calendar
repair_worker = run_repair
