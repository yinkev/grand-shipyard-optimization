from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

_WORKER_BOOT = (
    __name__ == "__main__"
    and len(sys.argv) > 1
    and sys.argv[1] == "__worker__"
)
if not _WORKER_BOOT:
    import candidate19_core as core
else:
    core = None
import c19_protocol as protocol


TELEMETRY: dict[str, object] = {}
_LAST_CHECK: dict[str, object] = {}
_ROLE_ORDER = {
    "floor": 0,
    "constructor": 1,
    "calendar": 2,
    "repair": 3,
    "repair-calendar": 3,
    "repair-global": 4,
    "c2pm": 4,
    "emergency": 4,
}
_ALL_IN_PRESSURE_MIN = 0.70
_ALL_IN_N_MIN = 250


def _discover_cpus() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        try:
            return tuple(sorted(int(cpu) for cpu in os.sched_getaffinity(0)))
        except Exception:
            pass
    return tuple(range(max(1, int(os.cpu_count() or 1))))


def _check(prob: dict, solution: dict | None) -> dict | None:
    _LAST_CHECK.clear()
    if not isinstance(solution, dict):
        _LAST_CHECK["error"] = "solution-not-dict"
        return None
    try:
        import utils

        report = utils.check_feasibility(prob, solution)
    except BaseException as exc:
        _LAST_CHECK["error"] = f"{type(exc).__name__}:{str(exc)[:160]}"
        return None
    if isinstance(report, dict):
        _LAST_CHECK.update(report)
    if not report or not report.get("feasible"):
        return None
    if report.get("objective") is None:
        return None
    return report


def _emergency(prob: dict) -> tuple[dict, dict] | None:
    try:
        solution = core._l1_sparse(prob)
    except BaseException:
        return None
    report = _check(prob, solution)
    if report is None:
        return None
    return solution, report


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    try:
        import utils  # type: ignore

        utils_dir = os.path.dirname(os.path.abspath(utils.__file__))
        old = env.get("PYTHONPATH", "")
        entries = [utils_dir]
        entries.extend(
            value
            for value in old.split(os.pathsep)
            if value and os.path.abspath(value) != utils_dir
        )
        env["PYTHONPATH"] = os.pathsep.join(entries)
    except Exception:
        pass
    return env


def _worker_command(
    role: str,
    problem_path: Path,
    root: Path,
    deadline: float,
    cpus: tuple[int, ...],
) -> list[str]:
    return [
        sys.executable,
        "-B",
        os.path.abspath(__file__),
        "__worker__",
        role,
        os.fspath(problem_path),
        os.fspath(root),
        repr(float(deadline)),
        ",".join(str(int(cpu)) for cpu in cpus),
    ]


def _spawn_worker(
    role: str,
    problem_path: Path,
    root: Path,
    deadline: float,
    cpus: tuple[int, ...],
) -> subprocess.Popen:
    return subprocess.Popen(
        _worker_command(role, problem_path, root, deadline, cpus),
        env=_child_env(),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _children_by_parent() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    proc = Path("/proc")
    if proc.is_dir():
        try:
            for entry in proc.iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    raw = (entry / "stat").read_text(encoding="utf-8")
                    tail = raw[raw.rfind(")") + 2 :].split()
                    pid = int(entry.name)
                    ppid = int(tail[1])
                    children.setdefault(ppid, []).append(pid)
                except Exception:
                    continue
            return children
        except Exception:
            children.clear()
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,ppid="],
            text=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
        for line in output.splitlines():
            fields = line.split()
            if len(fields) == 2:
                children.setdefault(int(fields[1]), []).append(int(fields[0]))
    except Exception:
        pass
    return children


def _descendants(pid: int) -> set[int]:
    children = _children_by_parent()
    found: set[int] = set()
    stack = list(children.get(int(pid), ()))
    while stack:
        child = stack.pop()
        if child in found:
            continue
        found.add(child)
        stack.extend(children.get(child, ()))
    return found


def _signal_tree(pid: int, sig: int) -> None:
    pids = _descendants(pid) | {int(pid)}
    current_group = os.getpgrp()
    groups: set[int] = set()
    for target in pids:
        try:
            group = os.getpgid(target)
        except (ProcessLookupError, PermissionError, OSError):
            continue
        if group != current_group:
            groups.add(group)
    for group in groups:
        try:
            os.killpg(group, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    for target in sorted(pids, reverse=True):
        if target == os.getpid():
            continue
        try:
            os.kill(target, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _reap(proc: subprocess.Popen | None, grace: float = 0.08) -> None:
    if proc is None:
        return
    if proc.poll() is None or _descendants(proc.pid):
        _signal_tree(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=max(0.01, grace))
        except subprocess.TimeoutExpired:
            _signal_tree(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=0.20)
    except Exception:
        _signal_tree(proc.pid, signal.SIGKILL)
        try:
            proc.wait(timeout=0.05)
        except Exception:
            pass


def _reap_all(
    workers: dict[str, subprocess.Popen],
    budget_s: float = 0.30,
) -> None:
    """Stop every process group concurrently inside one bounded window."""
    live = [proc for proc in workers.values() if proc.poll() is None]
    for proc in live:
        _signal_tree(proc.pid, signal.SIGTERM)
    soft_deadline = time.monotonic() + max(0.02, 0.55 * budget_s)
    while time.monotonic() < soft_deadline:
        if all(proc.poll() is not None for proc in live):
            break
        time.sleep(0.005)
    for proc in live:
        if proc.poll() is None or _descendants(proc.pid):
            _signal_tree(proc.pid, signal.SIGKILL)
    hard_deadline = time.monotonic() + max(0.02, 0.45 * budget_s)
    for proc in live:
        remaining = max(0.0, hard_deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except Exception:
            pass


def _peak_estimate(prob: dict) -> int | None:
    try:
        estimate = core._armor_l2_estimate(prob)
        return int(estimate["estimated_peak_bytes"])
    except Exception:
        return None


def _runtime_roles(
    cpus: tuple[int, ...],
    peak: int | None,
    timelimit: float,
    pressure: float | None = None,
    n_blocks: int | None = None,
) -> list[tuple[str, tuple[int, ...]]]:
    available = tuple(cpus[:4])
    if len(available) < 2:
        return []
    if (pressure is not None
            and pressure >= _ALL_IN_PRESSURE_MIN
            and n_blocks is not None
            and n_blocks >= _ALL_IN_N_MIN):
        if len(available) >= 4:
            roles = [
                ("constructor", (available[0],)),
                ("calendar", (available[1],)),
                ("repair-calendar", (available[2],)),
                ("c2pm", (available[3],)),
            ]
            if peak is None or len(roles) * peak <= 14_000_000_000:
                return roles
    roles: list[tuple[str, tuple[int, ...]]] = [
        ("floor", available[:2]),
    ]
    if len(available) >= 3:
        roles.append(("calendar", (available[2],)))
    if len(available) >= 4:
        roles.append(("repair", (available[3],)))
    if peak is not None:
        model_count = 2 + max(0, len(roles) - 1)
        while len(roles) > 1 and model_count * peak > 14_000_000_000:
            roles.pop()
            model_count -= 1
    return roles


def _worker_main(argv: list[str]) -> int:
    if len(argv) != 8 or argv[1] != "__worker__":
        return 2
    role = argv[2]
    problem_path = argv[3]
    root = argv[4]
    deadline = float(argv[5])
    cpus = tuple(int(value) for value in argv[6].split(",") if value)
    # argv[7] is reserved for forward-compatible worker generation.  Older
    # package runners may omit it, so the command currently supplies "1".
    protocol.set_affinity(cpus)
    import c19_workers

    if role == "floor":
        report = c19_workers.run_floor(
            problem_path, root, deadline, cpus)
    elif role == "constructor":
        report = c19_workers.run_constructor(
            problem_path, root, deadline, cpus)
    elif role == "calendar":
        report = c19_workers.run_calendar(
            problem_path, root, deadline, cpus)
    elif role == "repair":
        report = c19_workers.run_repair(
            problem_path, root, deadline, cpus)
    elif role == "repair-calendar":
        report = c19_workers.run_repair(
            problem_path,
            root,
            deadline,
            cpus,
            prefer_calendar=True,
            seed_offset=32452843,
            role=role,
            operator_phase=0,
        )
    elif role == "repair-global":
        report = c19_workers.run_repair(
            problem_path,
            root,
            deadline,
            cpus,
            prefer_calendar=False,
            seed_offset=49979687,
            role=role,
            operator_phase=1,
        )
    elif role == "c2pm":
        report = c19_workers.run_c2pm(
            problem_path, root, deadline, cpus)
    else:
        return 2
    try:
        protocol.atomic_write_json(
            Path(root) / f"status-{role}.json",
            report if isinstance(report, dict) else {"result": report},
        )
    except Exception:
        pass
    return 0


def algorithm(prob_info, timelimit=60):
    started = time.monotonic()
    try:
        total = max(1.0, float(timelimit))
    except Exception:
        total = 60.0

    # Solver constructors attach caches and normalize records in place.  Keep
    # an untouched instance copy for every parent-side official check.
    validation_prob = json.loads(json.dumps(prob_info, separators=(",", ":")))
    execution_prob = json.loads(json.dumps(validation_prob, separators=(",", ":")))
    cpus = _discover_cpus()
    if not sys.platform.startswith("linux") or len(cpus) < 2:
        import submission8_floor

        return submission8_floor.algorithm(execution_prob, timelimit=total)

    hard_deadline = started + total
    offense_reserve = min(4.0, max(1.50, 1.0 + 0.015 * total))
    worker_deadline = hard_deadline - offense_reserve
    floor_deadline = hard_deadline - min(0.60, max(0.20, 0.04 * total))
    root = Path(tempfile.mkdtemp(prefix=".candidate19-"))
    problem_path = root / "problem.json"
    best_path = root / "best.json"
    history_dir = None
    workers: dict[str, subprocess.Popen] = {}
    emergency = None
    selected_solution = None
    selected_report = None
    selected_role = None
    worker_status: dict[str, object] = {}
    best_seen: dict[str, object] | None = None
    best_validation_ok = False
    best_validation_report: dict[str, object] | None = None
    check_remaining_s = None
    peak = _peak_estimate(
        json.loads(json.dumps(validation_prob, separators=(",", ":")))
    )
    pressure = None
    pressure_deadline = min(
        worker_deadline,
        time.monotonic() + min(1.0, max(0.10, 0.02 * total)),
    )
    try:
        pressure = core._constructor_pressure_ratio(
            json.loads(json.dumps(validation_prob, separators=(",", ":"))),
            pressure_deadline,
        )
        if pressure is not None:
            pressure = float(pressure)
    except BaseException:
        pressure = None
    n_blocks = len(validation_prob.get("blocks") or ())
    roles = _runtime_roles(cpus, peak, total, pressure, n_blocks)
    all_in = bool(roles and roles[0][0] == "constructor")
    try:
        problem_path.write_text(
            json.dumps(validation_prob, separators=(",", ":")),
            encoding="utf-8",
        )
        emergency = _emergency(execution_prob)
        if emergency is None:
            remaining = max(0.5, worker_deadline - time.monotonic())
            try:
                return core._scored_full_armored_algorithm(
                    execution_prob,
                    timelimit=remaining,
                )
            except BaseException:
                return core._l1_sparse(execution_prob)
        solution, report = emergency
        protocol.publish_best(
            best_path,
            solution,
            float(report["objective"]),
            "emergency",
            history_dir=history_dir,
            metadata={"stage": "emergency"},
        )

        for role, worker_cpus in roles:
            role_deadline = floor_deadline if role == "floor" else worker_deadline
            workers[role] = subprocess.Popen(
                _worker_command(
                    role,
                    problem_path,
                    root,
                    role_deadline,
                    worker_cpus,
                )
                + ["1"],
                env=_child_env(),
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        while time.monotonic() < floor_deadline:
            if workers and all(proc.poll() is not None for proc in workers.values()):
                break
            time.sleep(0.01)
        _reap_all(workers)
        for role in workers:
            status = protocol.read_json(root / f"status-{role}.json")
            if status is not None:
                worker_status[role] = status

        best = protocol.read_best(best_path)
        if isinstance(best, dict):
            best_seen = {
                "generation": best.get("generation"),
                "objective": best.get("objective"),
                "source": best.get("source"),
            }
        if isinstance(best, dict) and time.monotonic() < hard_deadline - 0.25:
            check_remaining_s = hard_deadline - time.monotonic()
            report = _check(validation_prob, best.get("solution"))
            best_validation_report = dict(_LAST_CHECK)
            if report is not None:
                best_validation_ok = True
                selected_solution = best["solution"]
                selected_report = report
                selected_role = best.get("source")
        if selected_solution is None and emergency is not None:
            selected_solution, selected_report = emergency
            selected_role = "emergency"
    finally:
        _reap_all(workers, budget_s=0.08)
        exit_codes = {role: proc.poll() for role, proc in workers.items()}
        try:
            shutil.rmtree(root)
        except Exception:
            pass

    TELEMETRY.clear()
    TELEMETRY.update(
        {
            "candidate19": True,
            "candidate20": True,
            "candidate23": True,
            "architecture": (
                "pressure-all-in-constructor+calendar+repair+c2pm"
                if all_in else
                "exact-s8-floor+calendar+repair"
            ),
            "pressure": pressure,
            "all_in_pressure_min": _ALL_IN_PRESSURE_MIN,
            "all_in_n_min": _ALL_IN_N_MIN,
            "cpus": cpus,
            "roles": {role: list(worker_cpus) for role, worker_cpus in roles},
            "peak_estimate_bytes": peak,
            "selected_role": selected_role,
            "selected_objective": (
                selected_report.get("objective") if selected_report else None
            ),
            "worker_exit_codes": exit_codes,
            "worker_status": worker_status,
            "best_seen": best_seen,
            "best_validation_ok": best_validation_ok,
            "best_validation_report": best_validation_report,
            "check_remaining_s": check_remaining_s,
            "wall_s": time.monotonic() - started,
        }
    )
    if selected_solution is not None:
        return selected_solution
    return emergency[0]


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv))
