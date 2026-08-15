"""Candidate 11 causal-cone joint space-time repair.

The safety shell and C²PM seed stay immutable.  This module owns the only
mutable offensive mechanism: repeatedly reopen a coherent set of committed
blocks, enumerate concrete residency alternatives against the frozen outside
background, solve the small exact selection problem, and retain only strict
checker-valid improvements.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - runtime image supplies numpy
    np = None


@dataclass(frozen=True)
class Column:
    cid: int
    bid: int
    bay: int
    oi: int
    x: int
    y: int
    e: int
    xd: int
    contact: float = 0.0
    incumbent: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.cid, "bid": self.bid, "bay": self.bay,
            "oi": self.oi, "x": self.x, "y": self.y,
            "e": self.e, "xd": self.xd, "contact": self.contact,
            "is_incumbent": self.incumbent,
        }


def _dur(block: dict) -> int:
    return max(1, int(math.ceil(float(block["processing_time"]) - 1e-6)))


def _rel(block: dict) -> int:
    return int(math.ceil(float(block["release_time"]) - 1e-6))


def snapshot(model) -> dict[int, tuple[int, int, int, int, int]]:
    return {
        int(bid): (int(rec[0]), int(rec[1]), int(rec[2]),
                   int(rec[3]), int(rec[4]))
        for bid, rec in model.committed.items()
    }


def restore(model, state: dict[int, Sequence[int]]) -> None:
    for bid in list(model.committed):
        model.uncommit(bid)
    for bid, rec in sorted(state.items(), key=lambda row: (int(row[1][4]), int(row[0]))):
        bay, oi, x, y, e = (int(v) for v in rec)
        if not model.commit(int(bid), oi, bay, x, y, e):
            raise RuntimeError(f"joint-repair restore failed for block {bid}")


def replay_solution(model, solution: dict) -> None:
    """Replay an emitted checker solution into an empty model."""
    if model.committed:
        raise ValueError("replay requires an empty model")
    ops = solution.get("operations", {}) if isinstance(solution, dict) else {}
    seen = set()
    for day_raw in sorted(ops, key=lambda value: int(value)):
        day = int(day_raw)
        for op in ops[day_raw]:
            if str(op.get("type", "")).upper() != "ENTRY":
                continue
            bid = int(op["block_id"])
            if bid in seen:
                raise ValueError(f"duplicate ENTRY for block {bid}")
            seen.add(bid)
            if not model.commit(
                    bid, int(op["orient_idx"]), int(op["bay_id"]),
                    int(op["x"]), int(op["y"]), day):
                raise ValueError(f"replay commit failed for block {bid}")
    if len(seen) != int(model.n):
        raise ValueError(f"incomplete replay {len(seen)}/{model.n}")


def _loss_score(model, bid: int, master_entries: Sequence[int | None] | None) -> float:
    bay, _oi, _x, _y, e, xd = model.committed[bid]
    block = model.blocks[bid]
    tardy = max(0.0, float(xd) - float(block["due_date"]))
    prefs = block["bay_preferences"]
    pref = max(prefs) - prefs[bay]
    target = e
    if master_entries is not None and bid < len(master_entries):
        raw = master_entries[bid]
        if raw is not None:
            target = int(raw)
    shift = max(0, int(e) - int(target))
    return (float(model.w1) * (tardy + 0.75 * shift)
            + float(model.w3) * pref
            + 0.05 * float(model.w2) * float(block.get("workload", 0.0)))


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _resident_candidates(model, bay: int, e: int, xd: int) -> list[int]:
    out = []
    for bid, rec in model.committed.items():
        if int(rec[0]) != int(bay):
            continue
        if _interval_overlap(int(rec[4]), int(rec[5]), int(e), int(xd)) > 0:
            out.append(int(bid))
    return out


def _rank_blockers(model, seed: int, target_e: int,
                   master_entries: Sequence[int | None] | None) -> list[int]:
    block = model.blocks[seed]
    dur = _dur(block)
    current = model.committed[seed]
    current_bay = int(current[0])
    prefs = block["bay_preferences"]
    preferred = max(range(model.m), key=lambda j: (prefs[j], -j))
    bays = [current_bay, preferred]
    for bay in sorted(range(model.m), key=lambda j: (-prefs[j], j)):
        if bay not in bays:
            bays.append(bay)
    ranked: dict[int, float] = {}
    for bay_rank, bay in enumerate(bays):
        for other in _resident_candidates(model, bay, target_e, target_e + dur):
            if other == seed:
                continue
            rec = model.committed[other]
            overlap = _interval_overlap(int(rec[4]), int(rec[5]), target_e, target_e + dur)
            score = 20.0 * overlap + 8.0 / (1 + bay_rank)
            score += 1e-6 * _loss_score(model, other, master_entries)
            if int(rec[4]) == int(target_e):
                score += 6.0
            ranked[other] = max(ranked.get(other, float("-inf")), score)
    return sorted(ranked, key=lambda bid: (-ranked[bid], bid))


def causal_cone(model, seed: int, size: int,
                master_entries: Sequence[int | None] | None) -> list[int]:
    """Coherent retroactive cone around a high-loss block.

    It includes residents of the seed's earlier target windows in its current
    and alternative bays, then recursively expands through the most relevant
    blockers.  This is deliberately a set destroy, not a one-block move.
    """
    size = max(2, min(int(size), int(model.n)))
    current_e = int(model.committed[seed][4])
    target = current_e
    if master_entries is not None and seed < len(master_entries):
        raw = master_entries[seed]
        if raw is not None:
            target = max(_rel(model.blocks[seed]), min(current_e, int(raw)))
    if target >= current_e:
        target = max(_rel(model.blocks[seed]), current_e - 1)
    chosen = [int(seed)]
    chosen_set = {int(seed)}
    frontier = [(int(seed), int(target))]
    while frontier and len(chosen) < size:
        owner, owner_target = frontier.pop(0)
        for other in _rank_blockers(model, owner, owner_target, master_entries):
            if other in chosen_set:
                continue
            chosen.append(other)
            chosen_set.add(other)
            other_e = int(model.committed[other][4])
            other_target = other_e
            if master_entries is not None and other < len(master_entries):
                raw = master_entries[other]
                if raw is not None:
                    other_target = max(_rel(model.blocks[other]), min(other_e, int(raw)))
            frontier.append((other, other_target))
            if len(chosen) >= size:
                break
    if len(chosen) < size:
        remaining = [
            bid for bid in model.committed if bid not in chosen_set
        ]
        remaining.sort(key=lambda bid: (-_loss_score(model, bid, master_entries), bid))
        chosen.extend(remaining[:size - len(chosen)])
    return chosen[:size]


def _congested_windows(model, master_entries=None) -> list[tuple[float, int, int]]:
    days: dict[tuple[int, int], list[int]] = {}
    for bid, rec in model.committed.items():
        bay, _oi, _x, _y, e, xd = rec
        for day in range(int(e), int(xd)):
            days.setdefault((int(bay), day), []).append(int(bid))
    rows = []
    for (bay, day), bids in days.items():
        if len(bids) < 2:
            continue
        loss = sum(_loss_score(model, bid, master_entries) for bid in bids)
        rows.append((float(len(bids)) + 1e-7 * loss, bay, day))
    rows.sort(reverse=True)
    return rows


def spacetime_cone(model, size: int, master_entries=None,
                   pick_index: int = 0) -> list[int]:
    size = max(2, min(int(size), int(model.n)))
    windows = _congested_windows(model, master_entries)
    if not windows:
        bids = sorted(model.committed,
                      key=lambda bid: (-_loss_score(model, bid, master_entries), bid))
        return bids[:size]
    _, bay, day = windows[int(pick_index) % min(12, len(windows))]
    half = max(2, min(10, size // 4))
    rows = []
    for bid, rec in model.committed.items():
        if int(rec[0]) != bay:
            continue
        overlap = _interval_overlap(int(rec[4]), int(rec[5]), day - half, day + half + 1)
        if overlap <= 0:
            continue
        rows.append((
            -int(int(rec[4]) <= day < int(rec[5])),
            -overlap,
            -_loss_score(model, int(bid), master_entries),
            int(bid),
        ))
    rows.sort()
    chosen = [row[-1] for row in rows[:size]]
    if len(chosen) < size:
        rest = [bid for bid in sorted(model.committed) if bid not in set(chosen)]
        rest.sort(key=lambda bid: (-_loss_score(model, bid, master_entries), bid))
        chosen.extend(rest[:size - len(chosen)])
    return chosen[:size]


def preference_cone(model, size: int, master_entries=None) -> list[int]:
    size = max(2, min(int(size), int(model.n)))
    scored = []
    for bid, rec in model.committed.items():
        bay = int(rec[0])
        prefs = model.blocks[bid]["bay_preferences"]
        regret = max(prefs) - prefs[bay]
        scored.append((float(regret), _loss_score(model, bid, master_entries), -bid, bid))
    scored.sort(reverse=True)
    seed = scored[0][-1]
    preferred = max(range(model.m), key=lambda j: (
        model.blocks[seed]["bay_preferences"][j], -j))
    e, xd = int(model.committed[seed][4]), int(model.committed[seed][5])
    chosen = [seed]
    for bid in _resident_candidates(model, preferred, e, xd):
        if bid != seed and bid not in chosen:
            chosen.append(bid)
        if len(chosen) >= size:
            return chosen
    for row in scored:
        bid = row[-1]
        if bid not in chosen:
            chosen.append(bid)
        if len(chosen) >= size:
            break
    return chosen[:size]


def _anchor_options(model, bid: int, oi: int, bay: int, entry: int,
                    max_n: int = 2) -> list[tuple[int, int, float]]:
    mapped = model.feasible_map(bid, oi, bay, entry)
    if mapped is None or np is None:
        return []
    feasible, x_lo, y_lo = mapped
    ys, xs = np.nonzero(feasible)
    if len(ys) == 0:
        return []
    ax = xs.astype(np.int64) + int(x_lo)
    ay = ys.astype(np.int64) + int(y_lo)
    scores = np.zeros(len(ys), dtype=float)
    try:
        om = model.om(bid, oi)
        conv = model.contact_map(
            bid, oi, bay, entry, entry + _dur(model.blocks[bid]))
        scores = model.contact_at(conv, om, ax, ay)
    except Exception:
        pass
    first = min(range(len(ys)), key=lambda i: (
        -float(scores[i]), int(ay[i]), int(ax[i])))
    picked = [first]
    if max_n > 1 and len(ys) > 1:
        second = max(
            (i for i in range(len(ys)) if i != first),
            key=lambda i: (
                max(abs(int(ax[i]) - int(ax[first])),
                    abs(int(ay[i]) - int(ay[first]))),
                float(scores[i]), -int(ay[i]), -int(ax[i])))
        picked.append(second)
    return [(int(ax[i]), int(ay[i]), float(scores[i])) for i in picked[:max_n]]


def _local_cost(prob: dict, col: Column) -> float:
    block = prob["blocks"][col.bid]
    w = prob.get("weights", {})
    tardy = max(0.0, float(col.xd) - float(block["due_date"]))
    prefs = block["bay_preferences"]
    pref = max(prefs) - prefs[col.bay]
    return float(w.get("w1", 1.0)) * tardy + float(w.get("w3", 1.0)) * pref


def _date_candidates(block: dict, current_e: int, master_e: int | None,
                     long_mode: bool) -> list[int]:
    release = _rel(block)
    due_pull = int(math.floor(float(block["due_date"]) - _dur(block) + 1e-9))
    values = [release, due_pull, current_e - 4, current_e - 2,
              current_e - 1, current_e, current_e + 1]
    if master_e is not None:
        values.extend([int(master_e), int(master_e) - 1, int(master_e) + 1])
    if long_mode:
        values.extend([current_e - 12, current_e - 8, current_e + 2, current_e + 4])
    out = sorted({max(release, int(v)) for v in values})
    return out


def _generate_columns(prob: dict, model, victims: Sequence[int],
                      incumbent: dict[int, Sequence[int]],
                      master_entries: Sequence[int | None] | None,
                      deadline: float, long_mode: bool,
                      max_cols: int) -> tuple[list[Column], dict[int, list[int]], dict[str, Any]]:
    columns: list[Column] = []
    per_block: dict[int, list[int]] = {}
    cid = 0
    scanned = 0
    for bid in victims:
        if time.monotonic() >= deadline:
            break
        bay0, oi0, x0, y0, e0 = (int(v) for v in incumbent[bid])
        inc = Column(cid, bid, bay0, oi0, x0, y0, e0,
                     e0 + _dur(prob["blocks"][bid]), 0.0, True)
        columns.append(inc)
        per_block[bid] = [cid]
        cid += 1
        block = prob["blocks"][bid]
        master_e = None
        if master_entries is not None and bid < len(master_entries):
            raw = master_entries[bid]
            if raw is not None:
                master_e = int(raw)
        dates = _date_candidates(block, e0, master_e, long_mode)
        prefs = block["bay_preferences"]
        bays = [bay0]
        preferred = max(range(model.m), key=lambda j: (prefs[j], -j))
        if preferred not in bays:
            bays.append(preferred)
        for bay in sorted(range(model.m), key=lambda j: (-prefs[j], j)):
            if bay not in bays:
                bays.append(bay)
        orientations = [oi0] + [oi for oi in range(len(block["shape"])) if oi != oi0]
        alternatives: dict[tuple[int, int, int], list[Column]] = {}
        for entry in dates:
            for bay in bays:
                for oi in orientations:
                    if time.monotonic() >= deadline:
                        break
                    scanned += 1
                    for x, y, contact in _anchor_options(
                            model, bid, oi, bay, entry,
                            max_n=2 if long_mode else 1):
                        if (bay, oi, x, y, entry) == (bay0, oi0, x0, y0, e0):
                            continue
                        col = Column(-1, bid, bay, oi, x, y, entry,
                                     entry + _dur(block), contact, False)
                        alternatives.setdefault((bay, entry, oi), []).append(col)
                if time.monotonic() >= deadline:
                    break
            if time.monotonic() >= deadline:
                break
        # Keep objective-bearing diversity first, then one spatially diverse
        # witness per class.  Unlike FBRM, geometry-identical classes do not
        # consume the whole pool.
        ranked = []
        for key, values in alternatives.items():
            values.sort(key=lambda col: (
                _local_cost(prob, col), -col.contact,
                col.e, col.bay, col.oi, col.y, col.x))
            ranked.append(values[0])
            if long_mode and len(values) > 1:
                ranked.append(values[1])
        ranked.sort(key=lambda col: (
            _local_cost(prob, col), -col.contact,
            col.e, col.bay, col.oi, col.y, col.x))
        for raw in ranked[:max(0, int(max_cols) - 1)]:
            col = Column(cid, raw.bid, raw.bay, raw.oi, raw.x, raw.y,
                         raw.e, raw.xd, raw.contact, False)
            columns.append(col)
            per_block[bid].append(cid)
            cid += 1
    complete = len(per_block) == len(victims)
    counts = [len(per_block.get(bid, ())) for bid in victims]
    return columns, per_block, {
        "complete": complete, "scanned": scanned,
        "n_columns": len(columns),
        "alternative_blocks": sum(1 for count in counts if count > 1),
        "counts": counts,
    }


def _mask_overlap(om_a, mask_a, a: Column, om_b, mask_b, b: Column) -> bool:
    ar0, ac0 = a.y + om_a.oy, a.x + om_a.ox
    br0, bc0 = b.y + om_b.oy, b.x + om_b.ox
    r0, r1 = max(ar0, br0), min(ar0 + om_a.h, br0 + om_b.h)
    c0, c1 = max(ac0, bc0), min(ac0 + om_a.w, bc0 + om_b.w)
    if r0 >= r1 or c0 >= c1:
        return False
    aa = mask_a[r0-ar0:r1-ar0, c0-ac0:c1-ac0]
    bb = mask_b[r0-br0:r1-br0, c0-bc0:c1-bc0]
    return bool((aa & bb).any())


def _candidate_conflicts(model, a: Column, b: Column) -> bool:
    om_a, om_b = model.om(a.bid, a.oi), model.om(b.bid, b.oi)
    overlap = max(a.e, b.e) < min(a.xd, b.xd)
    a_event_inside_b = (b.e < a.e < b.xd) or (b.e < a.xd < b.xd)
    b_event_inside_a = (a.e < b.e < a.xd) or (a.e < b.xd < a.xd)
    for k in range(om_a.nlay):
        layer_a = om_a.layer[k]
        if overlap and k < om_b.nlay and _mask_overlap(
                om_a, layer_a, a, om_b, om_b.layer[k], b):
            return True
        if a_event_inside_b and k < om_b.nlay and _mask_overlap(
                om_a, layer_a, a, om_b, om_b.dsh[k], b):
            return True
        if b_event_inside_a:
            event = om_b.cum[min(k, om_b.nlay - 1)]
            if _mask_overlap(om_a, layer_a, a, om_b, event, b):
                return True
    return False


def pair_compatible(model, a: Column, b: Column) -> bool:
    if a.bid == b.bid or a.bay != b.bay:
        return True
    if a.xd <= b.e or b.xd <= a.e:
        return True
    if _candidate_conflicts(model, a, b) or _candidate_conflicts(model, b, a):
        return False
    if a.e == b.e:
        ka = (a.bid, a.oi, a.x, a.y)
        kb = (b.bid, b.oi, b.x, b.y)
        if model._obstructs(ka, kb) and model._obstructs(kb, ka):
            return False
    return True


def _build_conflicts(model, columns: Sequence[Column], deadline: float):
    conflicts: list[tuple[int, int]] = []
    tested = 0
    for pos, a in enumerate(columns):
        for b in columns[pos + 1:]:
            if a.bid == b.bid or a.bay != b.bay or a.xd <= b.e or b.xd <= a.e:
                continue
            if time.monotonic() >= deadline:
                return conflicts, tested, False
            tested += 1
            if not pair_compatible(model, a, b):
                conflicts.append((a.cid, b.cid))
    return conflicts, tested, True


def _solve_selection(prob: dict, victims: Sequence[int], columns: Sequence[Column],
                     per_block: dict[int, list[int]], conflicts,
                     no_goods: Sequence[Sequence[int]], time_limit: float,
                     cp_model) -> dict[str, Any]:
    if cp_model is None:
        return {"ok": False, "status": "NO_ORTOOLS"}
    by_id = {col.cid: col for col in columns}
    mdl = cp_model.CpModel()
    x = {cid: mdl.NewBoolVar(f"jr_{cid}") for cid in by_id}
    for bid in victims:
        ids = per_block.get(int(bid), [])
        if not ids:
            return {"ok": False, "status": "NO_COLUMNS"}
        mdl.AddExactlyOne([x[cid] for cid in ids])
    for a, b in conflicts:
        mdl.AddAtMostOne(x[int(a)], x[int(b)])
    for item in no_goods:
        ids = [int(cid) for cid in item if int(cid) in x]
        if ids:
            mdl.Add(sum(x[cid] for cid in ids) <= len(ids) - 1)

    blocks, bays = prob["blocks"], prob["bays"]
    weights = prob.get("weights", {})
    w1 = int(round(float(weights.get("w1", 1.0))))
    w2 = int(round(float(weights.get("w2", 1.0))))
    w3 = int(round(float(weights.get("w3", 1.0))))
    objective = []
    for cid, col in by_id.items():
        block = blocks[col.bid]
        tardy = max(0, col.xd - int(math.floor(float(block["due_date"]) + 1e-9)))
        prefs = block["bay_preferences"]
        pref = int(max(prefs) - prefs[col.bay])
        objective.append((w1 * tardy + w3 * pref) * x[cid])

    # Exact Z2 for the selected full cone plus the frozen background loads.
    fixed_loads = [0] * len(bays)
    # Background load is passed through synthetic fixed columns with cid<0? No:
    # the caller supplies it separately by embedding _fixed_loads in prob.
    raw_fixed = prob.get("_c11_fixed_loads")
    if isinstance(raw_fixed, list) and len(raw_fixed) == len(bays):
        fixed_loads = [int(round(float(v))) for v in raw_fixed]
    max_load = sum(int(round(float(b.get("workload", 0)))) for b in blocks)
    load_vars = []
    for bay in range(len(bays)):
        load = mdl.NewIntVar(0, max_load, f"jr_load_{bay}")
        terms: list[Any] = [fixed_loads[bay]]
        for cid, col in by_id.items():
            if col.bay == bay:
                wl = int(round(float(blocks[col.bid].get("workload", 0))))
                terms.append(wl * x[cid])
        mdl.Add(load == sum(terms))
        load_vars.append(load)
    if len(bays) >= 2 and w2:
        areas = [int(b["width"] * b["height"]) for b in bays]
        lcm = 1
        for area in areas:
            lcm = math.lcm(lcm, area)
        norms, bounds = [], []
        for bay, load in enumerate(load_vars):
            factor = lcm // areas[bay]
            bound = max_load * factor
            norm = mdl.NewIntVar(0, bound, f"jr_norm_{bay}")
            mdl.Add(norm == load * factor)
            norms.append(norm)
            bounds.append(bound)
        diffs = []
        mx = max(bounds, default=0)
        for a in range(len(bays)):
            for b in range(a + 1, len(bays)):
                delta = mdl.NewIntVar(-mx, mx, f"jr_delta_{a}_{b}")
                mdl.Add(delta == norms[a] - norms[b])
                diff = mdl.NewIntVar(0, mx, f"jr_diff_{a}_{b}")
                mdl.AddAbsEquality(diff, delta)
                diffs.append(diff)
        max_diff = mdl.NewIntVar(0, mx, "jr_max_diff")
        mdl.AddMaxEquality(max_diff, diffs)
        denom = len(bays) * lcm
        numerator = sum(areas)
        z2_bound = max(1, (numerator * mx) // denom + 1)
        z2 = mdl.NewIntVar(0, z2_bound, "jr_z2")
        mdl.Add(z2 * denom <= numerator * max_diff)
        mdl.Add(numerator * max_diff <= (z2 + 1) * denom - 1)
        objective.append(w2 * z2)
    mdl.Minimize(sum(objective))
    for bid in victims:
        for cid in per_block[int(bid)]:
            if by_id[cid].incumbent:
                mdl.AddHint(x[cid], 1)
                break
    if mdl.Validate():
        return {"ok": False, "status": "MODEL_INVALID"}
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.02, float(time_limit))
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 20260615
    start = time.perf_counter()
    status = solver.Solve(mdl)
    wall = time.perf_counter() - start
    ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if not ok:
        return {"ok": False, "status": solver.StatusName(status), "wall_s": wall}
    chosen = {}
    for bid in victims:
        for cid in per_block[int(bid)]:
            if solver.Value(x[cid]):
                chosen[int(bid)] = int(cid)
                break
    return {"ok": True, "status": solver.StatusName(status),
            "wall_s": wall, "chosen": chosen,
            "proxy_objective": float(solver.ObjectiveValue())}


def _materialize(model, prob: dict, victims: Sequence[int], columns: Sequence[Column],
                 chosen: dict[int, int], no_goods: list[list[int]],
                 deadline: float) -> tuple[bool, str]:
    by_id = {col.cid: col for col in columns}
    committed = []
    selected = [by_id[chosen[int(bid)]] for bid in victims]
    selected.sort(key=lambda col: (
        col.e, float(prob["blocks"][col.bid]["due_date"]),
        -float(prob["blocks"][col.bid].get("workload", 0)), col.bid))
    for col in selected:
        if time.monotonic() >= deadline:
            for bid in reversed(committed):
                if bid in model.committed:
                    model.uncommit(bid)
            return False, "deadline"
        if model.commit(col.bid, col.oi, col.bay, col.x, col.y, col.e):
            committed.append(col.bid)
            continue
        # Longer same-day cycles are not captured by pair conflicts. Exclude
        # the selected combination and let the restricted master choose again.
        no_goods.append([chosen[int(bid)] for bid in victims])
        for bid in reversed(committed):
            if bid in model.committed:
                model.uncommit(bid)
        return False, "cycle-or-commit"
    return True, "ok"


def _changed_dimensions(before: dict[int, Sequence[int]],
                        after: dict[int, Sequence[int]], victims: Sequence[int]) -> dict[str, int]:
    out = {"blocks": 0, "bay": 0, "orientation": 0, "anchor": 0, "entry": 0}
    for bid in victims:
        a, b = tuple(before[bid]), tuple(after[bid])
        if a != b:
            out["blocks"] += 1
        out["bay"] += int(a[0] != b[0])
        out["orientation"] += int(a[1] != b[1])
        out["anchor"] += int((a[2], a[3]) != (b[2], b[3]))
        out["entry"] += int(a[4] != b[4])
    return out


def _run_wave(prob: dict, model, victims: Sequence[int], master_entries,
              deadline: float, long_mode: bool, cp_model) -> dict[str, Any]:
    started = time.monotonic()
    incumbent = snapshot(model)
    baseline_obj = float(model.objective()[0])
    fixed_loads = list(model.loads)
    for bid in victims:
        if bid in model.committed:
            fixed_loads[int(model.committed[bid][0])] -= float(
                model.blocks[bid].get("workload", 0.0))
            model.uncommit(bid)
    work_prob = dict(prob)
    work_prob["_c11_fixed_loads"] = fixed_loads
    pool_deadline = min(deadline, started + (4.0 if long_mode else 2.0))
    max_cols = 12 if long_mode else 8
    columns, per_block, pool_tel = _generate_columns(
        work_prob, model, victims, incumbent, master_entries,
        pool_deadline, long_mode, max_cols)
    if not pool_tel["complete"] or any(len(per_block.get(bid, ())) == 0 for bid in victims):
        restore(model, incumbent)
        return {"ok": False, "reason": "pool-incomplete", "pool": pool_tel,
                "wall_s": time.monotonic() - started}
    conflicts, tested, complete = _build_conflicts(model, columns, pool_deadline)
    if not complete:
        restore(model, incumbent)
        return {"ok": False, "reason": "conflict-deadline", "pool": pool_tel,
                "conflict_tests": tested, "wall_s": time.monotonic() - started}
    no_goods: list[list[int]] = []
    solve_rows = []
    for _round in range(3):
        remain = deadline - time.monotonic()
        if remain <= 0.05:
            break
        solved = _solve_selection(
            work_prob, victims, columns, per_block, conflicts, no_goods,
            min(1.00 if long_mode else 0.75, max(0.02, remain - 0.02)), cp_model)
        solve_rows.append({k: v for k, v in solved.items() if k != "chosen"})
        if not solved.get("ok"):
            break
        ok, why = _materialize(
            model, work_prob, victims, columns, solved["chosen"], no_goods, deadline)
        if ok:
            candidate_obj = float(model.objective()[0])
            after = snapshot(model)
            changes = _changed_dimensions(incumbent, after, victims)
            if candidate_obj < baseline_obj - 1e-9:
                return {
                    "ok": True, "reason": "improved", "objective": candidate_obj,
                    "baseline_objective": baseline_obj, "changes": changes,
                    "pool": pool_tel, "conflict_tests": tested,
                    "n_conflicts": len(conflicts), "solve": solve_rows,
                    "wall_s": time.monotonic() - started,
                }
            restore(model, incumbent)
            return {
                "ok": False, "reason": "not-improved", "objective": candidate_obj,
                "baseline_objective": baseline_obj, "changes": changes,
                "pool": pool_tel, "conflict_tests": tested,
                "n_conflicts": len(conflicts), "solve": solve_rows,
                "wall_s": time.monotonic() - started,
            }
        # Materialization rolled selected blocks back; victims remain absent.
        if why != "cycle-or-commit":
            break
    restore(model, incumbent)
    return {"ok": False, "reason": "no-materialization", "pool": pool_tel,
            "conflict_tests": tested, "n_conflicts": len(conflicts),
            "solve": solve_rows, "wall_s": time.monotonic() - started}


def run_joint_repair(
        prob: dict,
        model,
        master_entries: Sequence[int | None] | None,
        deadline: float,
        cp_model,
        validate_fn: Callable[[dict, dict], dict | None],
        retain_fn: Callable[[dict], None] | None,
        role: str,
        generation_start: int,
        seed_objective: float,
        seed: int = 20260615,
        operator_phase: int = 0,
) -> dict[str, Any]:
    """Run the complete causal-cone joint repair until the role deadline."""
    started = time.monotonic()
    rng = random.Random(int(seed) + (0 if role == "nominal" else 7919))
    best_obj = float(seed_objective)
    best_state = snapshot(model)
    current_state = dict(best_state)
    generation = int(generation_start)
    waves = []
    accepts = 0
    validated = 0
    long_mode = (float(deadline) - started) >= 90.0
    max_layers = max(
        (sum(1 for layer in shape.get("layers", ()) if layer)
         for block in prob.get("blocks", ())
         for shape in block.get("shape", ())),
        default=1)
    # At 60 seconds, four-layer cells pay a much steeper geometry cost per
    # column.  Smaller coherent cones buy more complete joint decisions and
    # empirically preserve the useful causal operation.  Long budgets retain
    # the full macro range.
    sizes = ([12, 16, 24] if max_layers >= 3 and not long_mode
             else [12, 16, 24, 32, 48])
    if model.n < max(sizes):
        sizes = sorted({max(2, min(model.n, value)) for value in sizes})
    seed_order = sorted(
        model.committed,
        key=lambda bid: (-_loss_score(model, bid, master_entries), bid))
    wave_index = 0
    stagnation = 0
    operator_names = ("causal", "spacetime", "preference")
    operator_stats = {
        name: {"attempts": 0, "accepts": 0, "gain_fraction": 0.0,
               "wall_s": 0.0}
        for name in operator_names
    }
    # A wall deadline remains authoritative.  The iteration cap is a second
    # deterministic guard against any unexpectedly cheap/no-op wave spinning.
    max_waves = 4096 if long_mode else 512
    phase = int(operator_phase)
    while (wave_index < max_waves
           and time.monotonic() < float(deadline) - 0.15
           and best_obj > 0.0):
        remain = float(deadline) - time.monotonic()
        if remain < 0.35:
            break
        size = sizes[(wave_index + phase) % len(sizes)]
        if not long_mode and remain < 3.0:
            size = min(size, 16)
        if wave_index < len(operator_names):
            op_name = operator_names[
                (wave_index + phase) % len(operator_names)]
        elif wave_index % 9 == 8:
            # Deterministic exploration prevents early lucky noise from
            # permanently eliminating a structurally different operator.
            op_name = min(
                operator_names,
                key=lambda name: (operator_stats[name]["attempts"],
                                  operator_names.index(name)))
        else:
            def operator_score(name):
                stat = operator_stats[name]
                attempts = max(1, int(stat["attempts"]))
                rate = float(stat["gain_fraction"]) / max(
                    1e-6, float(stat["wall_s"]))
                exploration = 1e-5 / math.sqrt(attempts)
                success = float(stat["accepts"]) / attempts
                return (rate + exploration, success, -attempts,
                        -operator_names.index(name))
            op_name = max(operator_names, key=operator_score)
        op = operator_names.index(op_name)
        if op_name == "causal":
            seed_bid = seed_order[
                (operator_stats["causal"]["attempts"] + phase)
                % len(seed_order)]
            victims = causal_cone(model, seed_bid, size, master_entries)
        elif op_name == "spacetime":
            victims = spacetime_cone(
                model, size, master_entries,
                pick_index=operator_stats["spacetime"]["attempts"] + phase)
        else:
            victims = preference_cone(model, size, master_entries)
        # Long-budget macro ruin after sustained failure; never displaces the
        # protected best because each wave has an exact rollback snapshot.
        if long_mode and stagnation >= 12:
            target = max(size, int(math.ceil(0.20 * model.n)))
            ranked = sorted(model.committed,
                            key=lambda bid: (-_loss_score(model, bid, master_entries), bid))
            head = ranked[:max(1, target // 2)]
            tail = [bid for bid in model.committed if bid not in set(head)]
            rng.shuffle(tail)
            victims = (head + tail)[:target]
            op_name = "macro"
            stagnation = 0
        # A wave receives enough wall to express the full mechanism but cannot
        # monopolize the role. At long horizons it may spend more on its pool.
        wave_cap = min(remain - 0.10, 6.0 if long_mode else 3.0)
        result = _run_wave(
            prob, model, victims, master_entries,
            time.monotonic() + max(0.10, wave_cap), long_mode, cp_model)
        result["operator"] = op_name
        result["size"] = len(victims)
        pool_tel = result.get("pool") or {}
        compact = {
            "operator": op_name,
            "size": len(victims),
            "ok": bool(result.get("ok")),
            "reason": result.get("reason"),
            "objective": result.get("objective"),
            "baseline_objective": result.get("baseline_objective"),
            "changes": result.get("changes"),
            "wall_s": result.get("wall_s"),
            "n_columns": pool_tel.get("n_columns"),
            "alternative_blocks": pool_tel.get("alternative_blocks"),
            "n_conflicts": result.get("n_conflicts"),
            "solve_status": [row.get("status") for row in result.get("solve", ())],
        }
        waves.append(compact)
        if len(waves) > 256:
            del waves[:len(waves) - 256]
        fatal_solver = any(
            row.get("status") in ("NO_ORTOOLS", "MODEL_INVALID")
            for row in result.get("solve", ()))
        if op_name in operator_stats:
            stat = operator_stats[op_name]
            stat["attempts"] += 1
            stat["wall_s"] += max(0.0, float(result.get("wall_s") or 0.0))
            before = result.get("baseline_objective")
            after = result.get("objective")
            if result.get("ok") and before not in (None, 0) and after is not None:
                stat["accepts"] += 1
                stat["gain_fraction"] += max(
                    0.0, (float(before) - float(after)) / float(before))
        if fatal_solver:
            break
        if result.get("ok"):
            accepts += 1
            current_state = snapshot(model)
            solution = model.emit()
            report = validate_fn(prob, solution)
            validated += 1
            official = (float(report["objective"])
                        if report and report.get("feasible") and report.get("objective") is not None
                        else None)
            if official is not None and official < best_obj - 1e-9:
                best_obj = official
                best_state = dict(current_state)
                generation += 1
                if retain_fn is not None:
                    retain_fn({
                        "role": role,
                        "generation": generation,
                        "objective": best_obj,
                        "solution": solution,
                        "telemetry": {
                            "candidate11_joint_repair": True,
                            "accepted_operator": op_name,
                            "accepted_size": len(victims),
                            "changes": result.get("changes"),
                            "wave_index": wave_index,
                            "wall_s": time.monotonic() - started,
                        },
                    })
                stagnation = 0
            else:
                restore(model, best_state)
                current_state = dict(best_state)
                stagnation += 1
        else:
            restore(model, current_state)
            stagnation += 1
        wave_index += 1
    restore(model, best_state)
    solution = model.emit()
    return {
        "solution": solution,
        "objective": best_obj,
        "generation": generation,
        "telemetry": {
            "candidate11_joint_repair": True,
            "waves": waves,
            "n_waves": wave_index,
            "stored_wave_rows": len(waves),
            "operator_stats": operator_stats,
            "accepted_waves": accepts,
            "validated_improvements": validated,
            "long_mode": long_mode,
            "wall_s": time.monotonic() - started,
        },
    }
