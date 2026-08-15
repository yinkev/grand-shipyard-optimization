"""Small, process-safe primitives shared by Candidate 19 workers."""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


SCHEMA_VERSION = 1


def _as_path(path: os.PathLike[str] | str) -> Path:
    return Path(path)


@contextmanager
def _locked(path: Path, *, exclusive: bool) -> Iterator[None]:
    """Lock a stable companion inode; the JSON target itself is replaced."""
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(handle.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_write_json(
    path: os.PathLike[str] | str,
    payload: Any,
) -> None:
    """Durably publish one complete JSON value using same-directory replace."""
    target = _as_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                separators=(",", ":"),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json(
    path: os.PathLike[str] | str,
    default: Any = None,
) -> Any:
    """Read JSON, returning ``default`` for absent or incomplete/corrupt data."""
    try:
        with _as_path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return default


def _validated_objective(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("objective must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError("objective must be finite")
    return value


def _validated_best(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None
    if not isinstance(payload.get("generation"), int):
        return None
    try:
        _validated_objective(payload.get("objective"))
    except (TypeError, ValueError):
        return None
    if not isinstance(payload.get("source"), str):
        return None
    if not isinstance(payload.get("solution"), dict):
        return None
    return payload


def read_best(path: os.PathLike[str] | str) -> dict[str, Any] | None:
    """Read one complete incumbent while sharing the publisher lock."""
    target = _as_path(path)
    with _locked(target, exclusive=False):
        return _validated_best(read_json(target))


def publish_best(
    best_path: os.PathLike[str] | str,
    solution: Mapping[str, Any],
    objective: int | float,
    source: str,
    *,
    history_dir: os.PathLike[str] | str | None = None,
    metadata: Mapping[str, Any] | None = None,
    replace_equal: bool = False,
) -> dict[str, Any] | None:
    """Publish only a strictly better complete incumbent.

    The caller supplies an objective returned by the official checker. This
    function deliberately does not recompute it; it serializes the already
    checked solution and arbitrates concurrent publishers.
    """
    target = _as_path(best_path)
    checked_objective = _validated_objective(objective)
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    if not isinstance(solution, Mapping):
        raise TypeError("solution must be a mapping")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")

    # Fail before taking the publication lock if the solution cannot represent
    # one complete ENTRY/EXIT record per block.
    frozen_solution = json.loads(
        json.dumps(solution, allow_nan=False, separators=(",", ":"))
    )
    parse_solution_records(frozen_solution)

    with _locked(target, exclusive=True):
        incumbent = _validated_best(read_json(target))
        if incumbent is not None:
            if checked_objective > incumbent["objective"]:
                return None
            if (
                checked_objective == incumbent["objective"]
                and not replace_equal
            ):
                return None

        generation = 1 if incumbent is None else incumbent["generation"] + 1
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "generation": generation,
            "objective": checked_objective,
            "source": source,
            "solution": frozen_solution,
            "metadata": dict(metadata or {}),
            "published_ns": time.time_ns(),
        }
        atomic_write_json(target, payload)
        if history_dir is not None:
            history = _as_path(history_dir)
            history.mkdir(parents=True, exist_ok=True)
            atomic_write_json(history / f"{generation:08d}.json", payload)
        return payload


def list_history(
    history_dir: os.PathLike[str] | str,
) -> list[dict[str, Any]]:
    """Return valid generation snapshots in increasing generation order."""
    directory = _as_path(history_dir)
    if not directory.is_dir():
        return []
    result: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        payload = _validated_best(read_json(path))
        if payload is not None:
            result.append(payload)
    result.sort(key=lambda item: item["generation"])
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer")
    return parsed


def parse_solution_records(
    solution: Mapping[str, Any],
) -> dict[int, dict[str, int]]:
    """Parse a checker-shaped solution into one placement record per block."""
    if not isinstance(solution, Mapping):
        raise ValueError("solution must be a mapping")
    operations = solution.get("operations")
    if not isinstance(operations, Mapping):
        raise ValueError("solution.operations must be a mapping")

    records: dict[int, dict[str, int]] = {}
    entries: set[int] = set()
    exits: set[int] = set()
    for raw_day, day_operations in operations.items():
        day = _integer(raw_day, "operation day")
        if not isinstance(day_operations, list):
            raise ValueError(f"operations[{raw_day!r}] must be a list")
        for operation in day_operations:
            if not isinstance(operation, Mapping):
                raise ValueError("operation must be a mapping")
            kind = operation.get("type")
            if kind not in {"ENTRY", "EXIT"}:
                raise ValueError(f"unsupported operation type: {kind!r}")
            block_id = _integer(operation.get("block_id"), "block_id")
            record = records.setdefault(block_id, {"block_id": block_id})
            if kind == "ENTRY":
                if block_id in entries:
                    raise ValueError(f"duplicate ENTRY for block {block_id}")
                entries.add(block_id)
                record.update(
                    {
                        "entry": day,
                        "bay_id": _integer(operation.get("bay_id"), "bay_id"),
                        "x": _integer(operation.get("x"), "x"),
                        "y": _integer(operation.get("y"), "y"),
                        "orient_idx": _integer(
                            operation.get("orient_idx"),
                            "orient_idx",
                        ),
                    }
                )
            else:
                if block_id in exits:
                    raise ValueError(f"duplicate EXIT for block {block_id}")
                exits.add(block_id)
                record["exit"] = day

    missing_entry = sorted(exits - entries)
    missing_exit = sorted(entries - exits)
    if missing_entry or missing_exit:
        raise ValueError(
            "incomplete block operations: "
            f"missing ENTRY={missing_entry}, missing EXIT={missing_exit}"
        )
    return {block_id: records[block_id] for block_id in sorted(records)}


def _cpu_tuple(cpus: Iterable[int] | str) -> tuple[int, ...]:
    if isinstance(cpus, str):
        values: Iterable[Any] = (
            value.strip() for value in cpus.split(",") if value.strip()
        )
    else:
        values = cpus
    result: list[int] = []
    for value in values:
        cpu = _integer(value, "cpu")
        if cpu < 0:
            raise ValueError("cpu must be non-negative")
        if cpu not in result:
            result.append(cpu)
    return tuple(result)


def set_affinity(cpus: Iterable[int] | str) -> tuple[int, ...]:
    """Pin the current process and return the normalized requested CPU set."""
    requested = _cpu_tuple(cpus)
    if not requested:
        raise ValueError("at least one CPU is required")
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(requested))
    return requested
