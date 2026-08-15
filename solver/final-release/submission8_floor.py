# myalgorithm.py — OGC 2026, team Smoop
# cand_1q3: sol R8 ARCH E on cand_1q2 (04552bdb) — absolute 6.0s arm kill
# (no budget+1.25 wait), control >=52s @ tl=60, 2.0s setup/check/selection
# reserve; retains always-armored + feature-threshold λ + defensive telemetry
# + exception-hardened arm. Anytime ladder L1..L4; best utils-validated sol.

# Environment guard: must run before any numpy/scipy import we control (§6).
import gc
import glob
import importlib.util
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import hashlib
import json
import math
import random
import signal
import sys
import threading
import time
import tempfile

# Module-level incumbent holder: the except path must always be able to reach
# the best validated solution (§6). "solution" is ALWAYS utils-validated;
# "last_resort" is the unvalidated L1, returned only when nothing validated
# exists (the sole deliberate exception to validate-everything).
_INCUMBENT = {"solution": None, "objective": None, "obj1": None,
              "last_resort": None}
# Stop flags shared with the SIGALRM handler. "final" suppresses the raise
# once the function has entered its return path, so a late alarm cannot
# destroy a clean return.
_STOP = {"flag": False, "final": False}
# Telemetry for the dev harness (never printed; harmless on the server).
TELEMETRY = {}

_CKERNEL_MODULE_NAME = "_ogc2026_ckernel"
_CKERNEL_MODULE = None
_CKERNEL_LOAD_ATTEMPTED = False
_CKERNEL_LOAD_FAILURE = None

_THREADPOOL_LIMITER = None


def _reset_kernel_telemetry(route):
    TELEMETRY["kernel_loaded"] = False
    TELEMETRY["primitive_calls"] = 0
    TELEMETRY["kernel_calls"] = 0
    TELEMETRY["fallback_calls"] = 0
    TELEMETRY["route"] = route
    TELEMETRY["fallback_reason"] = None
    TELEMETRY["pressure_finish_calls"] = 0
    TELEMETRY["ff_obj_strict_changes"] = 0


def _load_ckernel():
    """Load a sibling extension without publishing a partial sys.modules entry."""
    global _CKERNEL_MODULE, _CKERNEL_LOAD_ATTEMPTED, _CKERNEL_LOAD_FAILURE
    if _CKERNEL_LOAD_ATTEMPTED:
        return _CKERNEL_MODULE
    _CKERNEL_LOAD_ATTEMPTED = True
    here = os.path.dirname(__file__)
    candidates = sorted(
        glob.glob(os.path.join(here, f"{_CKERNEL_MODULE_NAME}*.so")) +
        glob.glob(os.path.join(here, "..", "build",
                               f"{_CKERNEL_MODULE_NAME}*.so")))
    if not candidates:
        _CKERNEL_LOAD_FAILURE = "missing"
        return None
    previous = sys.modules.pop(_CKERNEL_MODULE_NAME, None)
    had_previous = previous is not None
    try:
        spec = importlib.util.spec_from_file_location(
            _CKERNEL_MODULE_NAME, candidates[0])
        if spec is None or spec.loader is None:
            _CKERNEL_LOAD_FAILURE = "load:missing-loader"
            if had_previous:
                sys.modules[_CKERNEL_MODULE_NAME] = previous
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[_CKERNEL_MODULE_NAME] = module
        _CKERNEL_MODULE = module
        return module
    except BaseException as exc:
        sys.modules.pop(_CKERNEL_MODULE_NAME, None)
        if had_previous:
            sys.modules[_CKERNEL_MODULE_NAME] = previous
        _CKERNEL_LOAD_FAILURE = f"load:{type(exc).__name__}"
        return None


_KERNEL_TELEMETRY_DEFAULTS = {
    "kernel_loaded": False,
    "primitive_calls": 0,
    "kernel_calls": 0,
    "fallback_calls": 0,
    "route": None,
    "fallback_reason": None,
    "pressure_finish_calls": 0,
    "ff_obj_strict_changes": 0,
}


def _finalize_kernel_telemetry():
    """Defensive: never KeyError if TELEMETRY was cleared (race parent path)."""
    values = {
        key: TELEMETRY.get(key, default)
        for key, default in _KERNEL_TELEMETRY_DEFAULTS.items()
    }
    TELEMETRY.clear()
    TELEMETRY.update(values)

_ARMOR_L2_RSS_CAP_BYTES = 8 * 1024 * 1024 * 1024
_ARMOR_L2_BASE_BYTES = 768 * 1024 * 1024


class _SelectorDeadline(Exception):
    pass


class _DeadlineAlarm(Exception):
    pass


def _cap_thread_pools():
    """Best-effort runtime cap on BLAS pools already initialized before our
    module-top env guard ran (utils -> shapely -> numpy import chain)."""
    global _THREADPOOL_LIMITER
    try:
        import threadpoolctl
        _THREADPOOL_LIMITER = threadpoolctl.threadpool_limits(limits=1)
    except Exception:
        pass


def _jitter_probe(duration=0.1):
    """Spin for `duration` seconds; return the max observed monotonic gap.
    Detects SIGSTOP/CONT throttling or heavy co-tenant load (§6)."""
    end = time.monotonic() + duration
    last = time.monotonic()
    gap = 0.0
    while True:
        now = time.monotonic()
        if now - last > gap:
            gap = now - last
        last = now
        if now >= end:
            break
    return gap


def _hidden_env_tuning_allowed():
    return os.environ.get("OGC_ALLOW_ENV_TUNING") == "1"


def _mark_validator_available():
    try:
        import utils  # noqa: F401
        TELEMETRY["validator_available"] = True
        return True
    except Exception:
        TELEMETRY["validator_available"] = False
        return False


def _rss_mb():
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024
    except Exception:
        return None


def _deadline_expired(deadline):
    if _STOP["flag"] or time.monotonic() >= deadline:
        raise _SelectorDeadline()


def _suppress_env(keys):
    saved = {}
    for key in keys:
        if key in os.environ:
            saved[key] = os.environ[key]
            del os.environ[key]
    return saved


def _restore_env(saved):
    for key, value in saved.items():
        os.environ[key] = value


# ---------------------------------------------------------------------------
# L1 sparse fallback (§6): every block alone in an empty-bay window of its
# best fitting bay. Provably feasible, terrible objective, ~tens of ms.
# ---------------------------------------------------------------------------

def _rel_day(blk):
    """Earliest integer entry day: utils compares entry >= release - 1e-6
    against the raw (possibly fractional) release_time."""
    return math.ceil(blk["release_time"] - 1e-6)


def _dur(blk):
    """Residency length: utils requires exit - entry >= P - 1e-6, and
    exit == entry is unencodable (§0.2.10), hence the max(., 1) clamp."""
    return max(math.ceil(blk["processing_time"] - 1e-6), 1)


def _placed_bbox(layers, x, y):
    """All-layers bbox after placement, replicating utils.Block exactly:
    drop empty layers (utils._resolve_layers), translate every layer by
    (x - ref_x, y - ref_y) where ref = first vertex of the first layer;
    bbox over all translated vertices. Zero-geometry blocks mirror utils'
    1x1 bounding_rect fallback."""
    layers = [l for l in layers if l]
    if not layers:
        return float(x), float(y), float(x) + 1.0, float(y) + 1.0
    ref_x, ref_y = layers[0][0]
    dx = x - ref_x
    dy = y - ref_y
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for layer in layers:
        for vx, vy in layer:
            wx = vx + dx
            wy = vy + dy
            if wx < min_x:
                min_x = wx
            if wx > max_x:
                max_x = wx
            if wy < min_y:
                min_y = wy
            if wy > max_y:
                max_y = wy
    return min_x, min_y, max_x, max_y


def _l1_find_spot(blk, bays):
    """First (bay, orient, x, y) where the block fits alone in an empty bay,
    bays tried in preference order. Placed-bbox fit test, ceil-snap aware
    (§0.2.7). Returns None only if the block fits nowhere (cannot happen on
    training-class inputs)."""
    prefs = blk["bay_preferences"]
    order = sorted(range(len(bays)), key=lambda j: (-prefs[j], j))
    for j in order:
        W = bays[j]["width"]
        H = bays[j]["height"]
        for oi, shape in enumerate(blk["shape"]):
            layers = [l for l in shape["layers"] if l]
            if not layers:
                if W >= 1 and H >= 1:
                    return j, oi, 0, 0
                continue
            ref_x, ref_y = layers[0][0]
            min_x = min_y = float("inf")
            max_x = max_y = float("-inf")
            for layer in layers:
                for vx, vy in layer:
                    if vx < min_x:
                        min_x = vx
                    if vx > max_x:
                        max_x = vx
                    if vy < min_y:
                        min_y = vy
                    if vy > max_y:
                        max_y = vy
            x0 = math.ceil(ref_x - min_x)
            y0 = math.ceil(ref_y - min_y)
            # ceil-snap can overflow by one cell on fractional bboxes: test the
            # exact placed bbox the way utils computes it before accepting.
            for x in (x0, x0 + 1):
                for y in (y0, y0 + 1):
                    bb = _placed_bbox(layers, x, y)
                    if bb[0] >= 0 and bb[1] >= 0 and bb[2] <= W and bb[3] <= H:
                        return j, oi, x, y
    return None


def _l1_sparse(prob_info):
    """Serialize blocks within each bay: non-overlapping residency windows,
    entry = max(release, previous exit), exit = entry + max(P, 1)."""
    bays = prob_info["bays"]
    blocks = prob_info["blocks"]
    by_bay = [[] for _ in bays]
    spots = {}
    for bid, blk in enumerate(blocks):
        try:
            spot = _l1_find_spot(blk, bays)
        except _DeadlineAlarm:
            raise
        except Exception:
            spot = None
        if spot is None:
            # Fits nowhere: emit at origin of the largest bay anyway (never
            # return nothing; this cell is already lost).
            j = max(range(len(bays)),
                    key=lambda i: bays[i]["width"] * bays[i]["height"])
            spot = (j, 0, 0, 0)
        spots[bid] = spot
        by_bay[spot[0]].append(bid)

    entries = {}   # day -> list of (bid, bay, x, y, oi)
    exits = {}     # day -> list of (bid, bay)
    schedule = {}  # bid -> (entry, exit)
    for j, bids in enumerate(by_bay):
        bids.sort(key=lambda b: (blocks[b]["release_time"],
                                 blocks[b]["due_date"], b))
        prev_exit = 0
        for bid in bids:
            blk = blocks[bid]
            e = max(_rel_day(blk), prev_exit)
            x = e + _dur(blk)
            prev_exit = x
            schedule[bid] = (e, x)
            _, oi, px, py = spots[bid]
            entries.setdefault(e, []).append((bid, j, px, py, oi))
            exits.setdefault(x, []).append((bid, j))
    return _emit(entries, exits)


def _l1_sparse_ordered(prob_info, order_key):
    bays = prob_info["bays"]
    blocks = prob_info["blocks"]
    by_bay = [[] for _ in bays]
    spots = {}
    for bid, blk in enumerate(blocks):
        try:
            spot = _l1_find_spot(blk, bays)
        except _DeadlineAlarm:
            raise
        except Exception:
            spot = None
        if spot is None:
            j = max(range(len(bays)),
                    key=lambda i: bays[i]["width"] * bays[i]["height"])
            spot = (j, 0, 0, 0)
        spots[bid] = spot
        by_bay[spot[0]].append(bid)

    entries = {}
    exits = {}
    schedule = {}
    for j, bids in enumerate(by_bay):
        bids.sort(key=lambda bid: order_key(blocks[bid], bid))
        prev_exit = 0
        for bid in bids:
            blk = blocks[bid]
            e = max(_rel_day(blk), prev_exit)
            x = e + _dur(blk)
            prev_exit = x
            schedule[bid] = (e, x)
            _, oi, px, py = spots[bid]
            entries.setdefault(e, []).append((bid, j, px, py, oi))
            exits.setdefault(x, []).append((bid, j))
    return _emit(entries, exits)


def _cheap_l1_improvement(prob_info, t0, timelimit, check_cost):
    if _INCUMBENT["solution"] is None:
        return
    now = time.monotonic()
    stop = min(now + min(1.0, 0.02 * float(timelimit)),
               t0 + float(timelimit) - (3 * check_cost + 0.5))
    if now >= stop:
        return
    keys = (
        lambda blk, bid: (blk["due_date"], blk["release_time"], bid),
        lambda blk, bid: (blk["due_date"] - blk["release_time"]
                          - blk["processing_time"], blk["due_date"], bid),
        lambda blk, bid: (-blk["workload"], blk["due_date"], bid),
    )
    for key in keys:
        if time.monotonic() >= stop:
            break
        try:
            sol = _l1_sparse_ordered(prob_info, key)
            if time.monotonic() + check_cost >= stop:
                break
            tc = time.monotonic()
            res = _validate(prob_info, sol)
            check_cost = time.monotonic() - tc
            if res is not None and res.get("feasible") and (
                    _INCUMBENT["objective"] is None or
                    res["objective"] < _INCUMBENT["objective"]):
                _INCUMBENT.update(solution=sol, objective=res["objective"],
                                  obj1=res.get("obj1"))
                TELEMETRY["armor_cheap_objective"] = res["objective"]
        except _DeadlineAlarm:
            raise
        except Exception:
            TELEMETRY["armor_cheap_errors"] = \
                TELEMETRY.get("armor_cheap_errors", 0) + 1


def _emit(entries, exits):
    """Build the operations dict: ascending integer day keys; within a day,
    ALL exits (every bay) before any entry (§0.2.8/.9)."""
    ops = {}
    for d in sorted(set(entries) | set(exits)):
        lst = []
        for bid, bay in exits.get(d, ()):
            lst.append({"type": "EXIT", "block_id": int(bid),
                        "bay_id": int(bay)})
        for bid, bay, x, y, oi in entries.get(d, ()):
            lst.append({"type": "ENTRY", "block_id": int(bid),
                        "bay_id": int(bay), "x": int(x), "y": int(y),
                        "orient_idx": int(oi)})
        ops[str(int(d))] = lst
    return {"operations": ops}


def _validate(prob_info, solution):
    """Ground-truth check via the organizers' utils.py. Returns the result
    dict, or None if utils is unavailable or itself crashes."""
    try:
        import utils
        return utils.check_feasibility(prob_info, solution)
    except _DeadlineAlarm:
        raise
    except Exception:
        return None


# ---------------------------------------------------------------------------
# L2 geometry + temporal kernel (§1, §2)
#
# Conservative per-cell occupancy: every mask is a superset of the polygon's
# true cell coverage, so mask-disjoint => polygon-interiors-disjoint =>
# utils-feasible. The temporal model fixes exit = entry + max(P,1) and
# enforces the event-feasibility invariant via EVT protection grids, so
# stages 2/3/5 hold by construction under arbitrary insertion order.
# ---------------------------------------------------------------------------

_np = None
_fftconvolve = None


def _lazy_imports():
    global _np, _fftconvolve
    if _np is None:
        import numpy
        from scipy.signal import fftconvolve
        _np = numpy
        _fftconvolve = fftconvolve


def _raster_layer(verts, ox, oy, h, w):
    """Conservative coverage mask of one anchored polygon over the cell
    window [ox, ox+w) x [oy, oy+h). A cell is flagged iff its center is
    inside the polygon OR an edge passes through its (closed) unit square —
    a superset of every cell whose open square meets the polygon interior."""
    np = _np
    n = len(verts)
    vx = np.array([v[0] for v in verts], dtype=np.float64)
    vy = np.array([v[1] for v in verts], dtype=np.float64)
    mask = np.zeros((h, w), dtype=bool)

    # Cell centers strictly inside (PNPOLY crossing parity, XOR per edge).
    X = ox + 0.5 + np.arange(w, dtype=np.float64)[None, :]
    Y = oy + 0.5 + np.arange(h, dtype=np.float64)[:, None]
    inside = np.zeros((h, w), dtype=bool)
    x2v = np.roll(vx, -1)
    y2v = np.roll(vy, -1)
    for i in range(n):
        x1, y1, x2, y2 = vx[i], vy[i], x2v[i], y2v[i]
        if y1 == y2:
            continue
        cross = ((y1 > Y) != (y2 > Y)) & \
                (X < (x2 - x1) * (Y - y1) / (y2 - y1) + x1)
        inside ^= cross
    mask |= inside

    # Supercover of every edge: enumerate grid-line crossings, take segment
    # midpoints between consecutive crossings, flag the containing cell.
    # The +/-1e-9 fan flags both neighbors when a midpoint sits exactly on a
    # grid line (e.g. axis-aligned edges on integer coordinates) — pure
    # over-flagging, which is the safe direction.
    for i in range(n):
        x1, y1, x2, y2 = vx[i], vy[i], x2v[i], y2v[i]
        dx = x2 - x1
        dy = y2 - y1
        ts = [np.array([0.0, 1.0])]
        if dx != 0.0:
            lo, hi = (x1, x2) if x1 < x2 else (x2, x1)
            ks = np.arange(math.ceil(lo), math.floor(hi) + 1, dtype=np.float64)
            ts.append((ks - x1) / dx)
        if dy != 0.0:
            lo, hi = (y1, y2) if y1 < y2 else (y2, y1)
            ks = np.arange(math.ceil(lo), math.floor(hi) + 1, dtype=np.float64)
            ts.append((ks - y1) / dy)
        tall = np.unique(np.concatenate(ts))
        tall = tall[(tall >= 0.0) & (tall <= 1.0)]
        if len(tall) < 2:
            continue
        mids = (tall[:-1] + tall[1:]) * 0.5
        mids = mids[(tall[1:] - tall[:-1]) > 1e-12]
        px = x1 + mids * dx
        py = y1 + mids * dy
        for sx in (-1e-9, 1e-9):
            ixs = np.floor(px + sx).astype(np.int64) - ox
            for sy in (-1e-9, 1e-9):
                iys = np.floor(py + sy).astype(np.int64) - oy
                ok = (ixs >= 0) & (ixs < w) & (iys >= 0) & (iys < h)
                mask[iys[ok], ixs[ok]] = True
    return mask


class _OrientMasks:
    """Per (block, orientation): aligned layer masks over one common cell
    window, plus derived families.
      layer[k] : bool mask of layer k
      cum[l]   : OR of layer[k] for k <= l           (EVT stamping)
      dsh[k]   : OR of layer[j] for j >= k           (shadow / obstruction)
      i16_*    : int16 0/1 copies for count-grid stamping
      bbox     : float all-layers bbox in the anchored frame
      (ox, oy) : cell-window offset; placement at (x, y) occupies world
                 cells [y+oy : y+oy+h, x+ox : x+ox+w] under the mask.
    """
    __slots__ = ("ox", "oy", "h", "w", "nlay", "layer", "cum", "dsh",
                 "i16_layer", "i16_cum", "i16_dsh", "bbox", "flip")

    def __init__(self, shape_entry):
        np = _np
        # utils._resolve_layers drops empty layers before indexing, so layer
        # indices k must be computed on the filtered list.
        layers_raw = [l for l in shape_entry["layers"] if l]
        if not layers_raw:
            z = np.zeros((1, 1), dtype=bool)
            self.bbox = (0.0, 0.0, 1.0, 1.0)
            self.ox = self.oy = 0
            self.w = self.h = self.nlay = 1
            self.layer = [z]
            self.cum = [z.copy()]
            self.dsh = [z.copy()]
            self.i16_layer = [z.astype(np.int16)]
            self.i16_cum = [self.cum[0].astype(np.int16)]
            self.i16_dsh = [self.dsh[0].astype(np.int16)]
            self.flip = [z.astype(np.float32)]
            return
        ref_x, ref_y = layers_raw[0][0]
        anch = [[(vx - ref_x, vy - ref_y) for vx, vy in layer]
                for layer in layers_raw]
        all_x = [p[0] for lay in anch for p in lay]
        all_y = [p[1] for lay in anch for p in lay]
        bminx, bmaxx = min(all_x), max(all_x)
        bminy, bmaxy = min(all_y), max(all_y)
        self.bbox = (bminx, bminy, bmaxx, bmaxy)
        self.ox = math.floor(bminx)
        self.oy = math.floor(bminy)
        self.w = max(1, math.ceil(bmaxx) - self.ox)
        self.h = max(1, math.ceil(bmaxy) - self.oy)
        self.nlay = len(anch)
        self.layer = [_raster_layer(av, self.ox, self.oy, self.h, self.w)
                      for av in anch]
        self.cum = []
        acc = None
        for m in self.layer:
            acc = m.copy() if acc is None else (acc | m)
            self.cum.append(acc)
        self.dsh = [None] * self.nlay
        acc = None
        for k in range(self.nlay - 1, -1, -1):
            acc = self.layer[k].copy() if acc is None else (acc | self.layer[k])
            self.dsh[k] = acc
        self.i16_layer = [m.astype(np.int16) for m in self.layer]
        self.i16_cum = [m.astype(np.int16) for m in self.cum]
        self.i16_dsh = [m.astype(np.int16) for m in self.dsh]
        # pre-flipped float32 kernels for FFT correlation
        self.flip = [m[::-1, ::-1].astype(np.float32) for m in self.layer]


class _Model:
    """Spatial/temporal model (§1, §2). Day-indexed count grids per
    (bay, level), grown on demand:
      RES[b][k][d] : residency counts, entry <= d < exit       (stage 4)
      SHA[b][l][d] : strict-straddle shadow counts, OR_{k>=l}  (stages 2/3)
      EVT[b][l][d] : event-protection counts                   (invariant)
    All stamping is symmetric +/-, so uncommit is exact."""

    def __init__(self, prob_info):
        _lazy_imports()
        self.prob = prob_info
        self.blocks = prob_info["blocks"]
        self.bays = [(int(b["width"]), int(b["height"]))
                     for b in prob_info["bays"]]
        self.n = len(self.blocks)
        self.m = len(self.bays)
        self.K = max(1, max((len(s["layers"]) for blk in self.blocks
                             for s in blk["shape"]), default=1))
        w = prob_info.get("weights", {})
        self.w1 = w.get("w1", 1.0)
        self.w2 = w.get("w2", 1.0)
        self.w3 = w.get("w3", 1.0)
        bay_areas = [W * H for W, H in self.bays]
        avg = sum(bay_areas) / self.m
        self.u = [avg / a for a in bay_areas]
        self._om = {}
        # Exact cache for forbidden grids while the committed model state is
        # unchanged. A placement scan evaluates many orientations against the
        # same (bay, layer, entry, exit) windows; commit/uncommit invalidates
        # every cached grid before the next state is queried.
        self._forbidden_cache = {}
        self.RES = [[{} for _ in range(self.K)] for _ in range(self.m)]
        self.SHA = [[{} for _ in range(self.K)] for _ in range(self.m)]
        self.EVT = [[{} for _ in range(self.K)] for _ in range(self.m)]
        self.committed = {}        # bid -> (bay, oi, x, y, e, xd)
        self.day_entries = {}      # (bay, day) -> [bid, ...]
        self.loads = [0.0] * self.m
        self._ckernel = _load_ckernel()
        self._ckernel_disabled = self._ckernel is None
        TELEMETRY["kernel_loaded"] = self._ckernel is not None
        if self._ckernel_disabled:
            TELEMETRY["fallback_reason"] = _CKERNEL_LOAD_FAILURE

    def om(self, bid, oi):
        key = (bid, oi)
        r = self._om.get(key)
        if r is None:
            r = _OrientMasks(self.blocks[bid]["shape"][oi])
            self._om[key] = r
        return r

    # -- grid plumbing ------------------------------------------------------
    def _grid(self, fam, b, l, d):
        g = fam[b][l].get(d)
        if g is None:
            W, H = self.bays[b]
            g = _np.zeros((H, W), dtype=_np.int16)
            fam[b][l][d] = g
        return g

    def _apply(self, bid, sign):
        b, oi, x, y, e, xd = self.committed[bid]
        om = self.om(bid, oi)
        r0, c0 = y + om.oy, x + om.ox
        rs, cs = slice(r0, r0 + om.h), slice(c0, c0 + om.w)
        for d in range(e, xd):
            for k in range(om.nlay):
                self._grid(self.RES, b, k, d)[rs, cs] += sign * om.i16_layer[k]
        for d in range(e + 1, xd):
            for l in range(om.nlay):
                self._grid(self.SHA, b, l, d)[rs, cs] += sign * om.i16_dsh[l]
        for t in (e, xd):
            for l in range(self.K):
                self._grid(self.EVT, b, l, t)[rs, cs] += \
                    sign * om.i16_cum[min(l, om.nlay - 1)]

    # -- candidate query (§1) -------------------------------------------------
    def anchor_range(self, om, b):
        """Integer placement ranges from the float all-layers bbox plus the
        raster window. Boundary cells are exact-verified again at commit."""
        W, H = self.bays[b]
        x_lo = max(math.ceil(-om.bbox[0]), -om.ox)
        x_hi = min(math.floor(W - om.bbox[2]), W - om.ox - om.w)
        y_lo = max(math.ceil(-om.bbox[1]), -om.oy)
        y_hi = min(math.floor(H - om.bbox[3]), H - om.oy - om.h)
        return x_lo, x_hi, y_lo, y_hi

    def forbidden(self, bid, oi, b, k, e, xd):
        """OR of the four §1 clauses for layer k of the candidate window.

        The result depends only on the committed model state and
        (bay, layer, entry, exit), not on block/orientation. Placement and
        repair scans therefore reuse it exactly across orientation calls.
        """
        key = (b, k, e, xd)
        cached = self._forbidden_cache.get(key)
        if cached is not None:
            TELEMETRY["forbidden_cache_hits"] = \
                TELEMETRY.get("forbidden_cache_hits", 0) + 1
            return cached
        TELEMETRY["forbidden_cache_misses"] = \
            TELEMETRY.get("forbidden_cache_misses", 0) + 1
        np = _np
        W, H = self.bays[b]
        F = np.zeros((H, W), dtype=bool)
        RESk = self.RES[b][k]
        for d in range(e, xd):
            g = RESk.get(d)
            if g is not None:
                F |= (g != 0)
        SHAk = self.SHA[b][k]
        for t in (e, xd):
            g = SHAk.get(t)
            if g is not None:
                F |= (g != 0)
        EVTk = self.EVT[b][k]
        for d in range(e + 1, xd):
            g = EVTk.get(d)
            if g is not None:
                F |= (g != 0)
        F.flags.writeable = False
        self._forbidden_cache[key] = F
        return F

    def feasible_map(self, bid, oi, b, e):
        """Bool grid over anchor positions: True = placement passes all four
        clauses for every layer. Returns (feas, x_lo, y_lo) where
        feas[y - y_lo, x - x_lo] indexes placements, or None if no anchor
        fits the bay at all."""
        np = _np
        om = self.om(bid, oi)
        W, H = self.bays[b]
        xd = e + _dur(self.blocks[bid])
        x_lo, x_hi, y_lo, y_hi = self.anchor_range(om, b)
        if x_lo > x_hi or y_lo > y_hi:
            return None
        TELEMETRY["primitive_calls"] += 1
        if self._ckernel is not None and not self._ckernel_disabled:
            if H <= 29 and W <= 179:
                forbidden = []
                masks = []
                for k in range(om.nlay):
                    F = self.forbidden(bid, oi, b, k, e, xd)
                    M = om.layer[k]
                    if (F.dtype != np.bool_ or M.dtype != np.bool_ or
                            not F.flags.c_contiguous or
                            not M.flags.c_contiguous):
                        TELEMETRY["fallback_reason"] = \
                            "contract:dtype-contiguity"
                        break
                    forbidden.append(F)
                    masks.append(M)
                else:
                    try:
                        result = self._ckernel.feasible_map_bitset(
                            forbidden, masks, H, W, om.h, om.w,
                            om.ox, om.oy, x_lo, x_hi, y_lo, y_hi)
                        TELEMETRY["kernel_calls"] += 1
                        return result
                    except _DeadlineAlarm:
                        raise
                    except Exception as exc:
                        self._ckernel_disabled = True
                        TELEMETRY["fallback_reason"] = \
                            f"exception:{type(exc).__name__}"
            else:
                TELEMETRY["fallback_reason"] = "contract:dimensions"
        TELEMETRY["fallback_calls"] += 1
        feas = np.ones((y_hi - y_lo + 1, x_hi - x_lo + 1), dtype=bool)
        for k in range(om.nlay):
            F = self.forbidden(bid, oi, b, k, e, xd)
            if not F.any():
                continue
            conv = _fftconvolve(F.astype(np.float32), om.flip[k], mode="full")
            sub = conv[om.h - 1:H, om.w - 1:W]   # indexed by (y+oy, x+ox)
            bad = sub > 0.5
            feas &= ~bad[y_lo + om.oy: y_hi + om.oy + 1,
                         x_lo + om.ox: x_hi + om.ox + 1]
        return feas, x_lo, y_lo

    # -- same-day precedence (§2) ---------------------------------------------
    def _obstructs(self, a_key, b_key):
        """True if A (present) may obstruct B's crane move: exists k with
        B.layer[k] overlapping A.dsh[k] (j >= k rule on conservative masks)."""
        a_bid, a_oi, ax, ay = a_key
        b_bid, b_oi, bx, by = b_key
        omA = self.om(a_bid, a_oi)
        omB = self.om(b_bid, b_oi)
        ar0, ac0 = ay + omA.oy, ax + omA.ox
        br0, bc0 = by + omB.oy, bx + omB.ox
        r0 = max(ar0, br0)
        r1 = min(ar0 + omA.h, br0 + omB.h)
        c0 = max(ac0, bc0)
        c1 = min(ac0 + omA.w, bc0 + omB.w)
        if r0 >= r1 or c0 >= c1:
            return False
        for k in range(min(omB.nlay, omA.nlay)):
            a = omA.dsh[k][r0 - ar0:r1 - ar0, c0 - ac0:c1 - ac0]
            bm = omB.layer[k][r0 - br0:r1 - br0, c0 - bc0:c1 - bc0]
            if (a & bm).any():
                return True
        return False

    def _key(self, bid):
        b, oi, x, y, e, xd = self.committed[bid]
        return (bid, oi, x, y)

    def _topo_day(self, bids, extra=None):
        """Topological entry order for one (bay, day): if A obstructs B's
        entry, B must precede A. Returns ordered list or None on cycle."""
        keys = {}
        for bid in bids:
            keys[bid] = self._key(bid)
        if extra is not None:
            keys[None] = extra
        ids = list(keys)
        succ = {u: [] for u in ids}
        indeg = {u: 0 for u in ids}
        for i, u in enumerate(ids):
            for v in ids[i + 1:]:
                # u present obstructs v moving => v before u
                if self._obstructs(keys[u], keys[v]):
                    succ[v].append(u)
                    indeg[u] += 1
                if self._obstructs(keys[v], keys[u]):
                    succ[u].append(v)
                    indeg[v] += 1
        import heapq
        heap = [(-1 if u is None else u) for u in ids if indeg[u] == 0]
        heapq.heapify(heap)
        order = []
        while heap:
            uu = heapq.heappop(heap)
            u = None if uu == -1 else uu
            order.append(u)
            for v in succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    heapq.heappush(heap, (-1 if v is None else v))
        if len(order) != len(ids):
            return None
        return order

    # -- commit / uncommit ----------------------------------------------------
    def commit(self, bid, oi, b, x, y, e):
        """Authoritative acceptance: exact placed-bbox verify (utils
        arithmetic), same-day precedence acyclicity, then stamp."""
        assert bid not in self.committed, f"double commit of block {bid}"
        self._forbidden_cache.clear()
        blk = self.blocks[bid]
        xd = e + _dur(blk)
        W, H = self.bays[b]
        bb = _placed_bbox(blk["shape"][oi]["layers"], x, y)
        if not (bb[0] >= 0 and bb[1] >= 0 and bb[2] <= W and bb[3] <= H):
            return False
        day = self.day_entries.get((b, e), [])
        if day and self._topo_day(day, extra=(bid, oi, x, y)) is None:
            return False
        self.committed[bid] = (b, oi, x, y, e, xd)
        self._apply(bid, +1)
        self.day_entries.setdefault((b, e), []).append(bid)
        self.loads[b] += blk["workload"]
        return True

    def uncommit(self, bid):
        self._forbidden_cache.clear()
        b, oi, x, y, e, xd = self.committed[bid]
        self._apply(bid, -1)
        del self.committed[bid]
        lst = self.day_entries[(b, e)]
        lst.remove(bid)
        if not lst:
            del self.day_entries[(b, e)]
        self.loads[b] -= self.blocks[bid]["workload"]

    # -- emission (§6) ----------------------------------------------------------
    def emit(self):
        """Operations dict: ascending integer day keys; per day ALL exits
        (within a bay latest-entrant-first), then all entries (per-bay
        precedence topo order)."""
        entry_rank = {}
        entries_by_day = {}
        exits_by_day = {}
        for (b, e), bids in sorted(self.day_entries.items(),
                                   key=lambda kv: (kv[0][1], kv[0][0])):
            order = self._topo_day(bids)
            entries_by_day.setdefault(e, []).extend(
                (b, bid) for bid in order)
            for seq, bid in enumerate(order):
                entry_rank[bid] = (e, seq)
        for bid, (b, oi, x, y, e, xd) in self.committed.items():
            exits_by_day.setdefault(xd, []).append((b, bid))
        ops = {}
        for d in sorted(set(entries_by_day) | set(exits_by_day)):
            lst = []
            for b, bid in sorted(exits_by_day.get(d, ()),
                                 key=lambda t: (t[0],) +
                                 tuple(-z for z in entry_rank[t[1]])):
                lst.append({"type": "EXIT", "block_id": int(bid),
                            "bay_id": int(b)})
            for b, bid in entries_by_day.get(d, ()):
                _, oi, x, y, _, _ = self.committed[bid]
                lst.append({"type": "ENTRY", "block_id": int(bid),
                            "bay_id": int(b), "x": int(x), "y": int(y),
                            "orient_idx": int(oi)})
            ops[str(int(d))] = lst
        return {"operations": ops}

    def z2_delta(self, b, workload):
        """Change in floor'd max pairwise normalized-load imbalance if
        `workload` is added to bay b. O(m^2), ordering use only."""
        if self.m < 2:
            return 0.0
        vals = [self.u[j] * self.loads[j] for j in range(self.m)]
        cur = max(abs(vals[i] - vals[j]) for i in range(self.m)
                  for j in range(self.m) if i != j)
        vals[b] += self.u[b] * workload
        new = max(abs(vals[i] - vals[j]) for i in range(self.m)
                  for j in range(self.m) if i != j)
        return math.floor(new) - math.floor(cur)

    def contact_map(self, bid, oi, b, e, xd):
        """Per-anchor packing score: count of occupied-or-wall cells 8-adjacent
        to the placed layer-0 mask (occupancy = residents during [e, xd)).
        Returns the (y, x)-anchored full correlation, or None if nothing to
        touch yet."""
        np = _np
        om = self.om(bid, oi)
        W, H = self.bays[b]
        U = np.ones((H + 2, W + 2), dtype=np.float32)  # 1-cell wall border
        U[1:-1, 1:-1] = 0.0
        RES0 = self.RES[b][0]
        inner = U[1:-1, 1:-1]
        for d in range(e, xd):
            g = RES0.get(d)
            if g is not None:
                inner += (g != 0)
        np.minimum(inner, 1.0, out=inner)
        m0 = om.layer[0]
        ker = np.zeros((om.h + 2, om.w + 2), dtype=np.float32)
        ker[0:om.h, 0:om.w] += m0
        ker[0:om.h, 1:om.w + 1] += m0
        ker[0:om.h, 2:om.w + 2] += m0
        ker[1:om.h + 1, 0:om.w] += m0
        ker[1:om.h + 1, 2:om.w + 2] += m0
        ker[2:om.h + 2, 0:om.w] += m0
        ker[2:om.h + 2, 1:om.w + 1] += m0
        ker[2:om.h + 2, 2:om.w + 2] += m0
        np.minimum(ker, 1.0, out=ker)
        conv = _fftconvolve(U, ker[::-1, ::-1], mode="full")
        # U is padded by 1 and ker by 1 relative to the mask window, so the
        # anchor (x, y) maps to conv[y+oy+h+1, x+ox+w+1] under 'full' indexing
        # offset (kh-1, kw-1) = (h+1, w+1) minus the pad shift (1, 1).
        return conv

    def contact_at(self, conv, om, xs, ys):
        return conv[ys + om.oy + om.h + 1, xs + om.ox + om.w + 1]

    # -- internal objective (search ordering only; promotion uses utils) -------
    def objective(self):
        obj1 = 0.0
        obj3 = 0.0
        for bid, (b, oi, x, y, e, xd) in self.committed.items():
            blk = self.blocks[bid]
            obj1 += max(0.0, float(xd) - blk["due_date"])
            prefs = blk["bay_preferences"]
            obj3 += max(prefs) - prefs[b]
        if self.m >= 2:
            vals = [self.u[j] * self.loads[j] for j in range(self.m)]
            obj2 = math.floor(max(abs(vals[i] - vals[j])
                                  for i in range(self.m)
                                  for j in range(self.m) if i != j))
        else:
            obj2 = 0.0
        return self.w1 * obj1 + self.w2 * obj2 + self.w3 * obj3, \
            obj1, obj2, obj3


# ---------------------------------------------------------------------------
# L3 constructor (§3): slack-priority greedy over the candidate machinery.
# ---------------------------------------------------------------------------

def _scored_cell(model, bid, oi, b, e, xd, xs, ys, x_lo, y_lo):
    """Choose the archived compact boundary cell for fallback placement."""
    TELEMETRY["scored_fallback_calls"] = (
        TELEMETRY.get("scored_fallback_calls", 0) + 1)
    np = _np
    om = model.om(bid, oi)
    conv = model.contact_map(bid, oi, b, e, xd)
    W, H = model.bays[b]
    axs = xs + x_lo
    ays = ys + y_lo
    contact = model.contact_at(conv, om, axs, ays)
    cxs = axs + (om.bbox[0] + om.bbox[2]) * 0.5
    cys = ays + (om.bbox[1] + om.bbox[3]) * 0.5
    d2 = (cxs - W * 0.5) ** 2 + (cys - H * 0.5) ** 2
    cell = contact + 0.5 * d2 / (W * W + H * H)
    i = int(np.argmax(cell))
    return int(axs[i]), int(ays[i])


def _place_block(model, bid, full_scan=False, scored_fallback=False):
    """Find and commit the best placement for one block. Returns True unless
    the block fits nowhere even via the empty-window fallback."""
    np = _np
    blk = model.blocks[bid]
    prefs = blk["bay_preferences"]
    pmax = max(prefs)
    R = _rel_day(blk)
    dur = _dur(blk)
    due = blk["due_date"]
    wl = blk["workload"]
    nor = len(blk["shape"])
    bay_order = sorted(range(model.m), key=lambda j: (-prefs[j], j))
    cand_bays = bay_order if full_scan else bay_order[:3]

    # (entry-window, bay) combos sorted by orientation-independent base cost;
    # a combo whose base exceeds the best found so far cannot win.
    combos = []
    for e in (R, R + 1, R + 2):
        tardy = max(0.0, float(e + dur) - due)
        for b in cand_bays:
            base = model.w1 * tardy + model.w3 * (pmax - prefs[b]) \
                + model.w2 * model.z2_delta(b, wl)
            combos.append((base, e, b))
    combos.sort()

    best = None   # (base, -cellscore, oi, b, x, y, e)
    for base, e, b in combos:
        if best is not None and base > best[0]:
            break
        xd = e + dur
        conv = None
        om0 = None
        for oi in range(nor):
            r = model.feasible_map(bid, oi, b, e)
            if r is None:
                continue
            feas, x_lo, y_lo = r
            ys, xs = np.nonzero(feas)
            if len(ys) == 0:
                continue
            om = model.om(bid, oi)
            conv = model.contact_map(bid, oi, b, e, xd)
            W, H = model.bays[b]
            axs = xs + x_lo
            ays = ys + y_lo
            contact = model.contact_at(conv, om, axs, ays)
            # keep-the-middle-clear tiebreak: weakly prefer cells far from
            # the bay center (normalized below 1 contact unit)
            cxs = axs + (om.bbox[0] + om.bbox[2]) * 0.5
            cys = ays + (om.bbox[1] + om.bbox[3]) * 0.5
            d2 = (cxs - W * 0.5) ** 2 + (cys - H * 0.5) ** 2
            cell = contact + 0.5 * d2 / (W * W + H * H)
            i = int(np.argmax(cell))
            cand = (base, -float(cell[i]), oi, b, int(axs[i]), int(ays[i]), e)
            if best is None or cand < best:
                best = cand
    if best is not None:
        _, _, oi, b, x, y, e = best
        if model.commit(bid, oi, b, x, y, e):
            return True
    if not full_scan:
        return _place_block(model, bid, full_scan=True,
                            scored_fallback=scored_fallback)

    # Day-by-day extension beyond R+2 while infeasible everywhere.
    for e in range(R + 3, R + 3 + 1000):
        for b in bay_order:
            for oi in range(nor):
                r = model.feasible_map(bid, oi, b, e)
                if r is None:
                    continue
                feas, x_lo, y_lo = r
                ys, xs = np.nonzero(feas)
                if len(ys) == 0:
                    continue
                if scored_fallback:
                    x, y = _scored_cell(model, bid, oi, b, e, e + dur,
                                        xs, ys, x_lo, y_lo)
                else:
                    x, y = int(xs[0] + x_lo), int(ys[0] + y_lo)
                if model.commit(bid, oi, b, x, y, e):
                    return True
        # cheap cap: jump to the empty-window fallback once past every
        # committed exit in every bay (nothing left to wait for)
        if all(e > max((c[5] for c in model.committed.values()
                        if c[0] == b2), default=0)
               for b2 in range(model.m)):
            break
    return _fast_finish_block(model, bid,
                              scored_fallback=scored_fallback)


def _fast_finish_block(model, bid, scored_fallback=False):
    """L1 mechanism inside the model: first empty-bay window of the best
    fitting bay (used by no-fit and repair deadline fallbacks)."""
    np = _np
    blk = model.blocks[bid]
    prefs = blk["bay_preferences"]
    R = _rel_day(blk)
    dur = _dur(blk)
    for b in sorted(range(model.m), key=lambda j: (-prefs[j], j)):
        e = max([R] + [c[5] for c in model.committed.values() if c[0] == b])
        for oi in range(len(blk["shape"])):
            r = model.feasible_map(bid, oi, b, e)
            if r is None:
                continue
            feas, x_lo, y_lo = r
            ys, xs = np.nonzero(feas)
            if len(ys) == 0:
                continue
            if scored_fallback:
                x, y = _scored_cell(model, bid, oi, b, e, e + dur,
                                    xs, ys, x_lo, y_lo)
            else:
                x, y = int(xs[0] + x_lo), int(ys[0] + y_lo)
            if model.commit(bid, oi, b, x, y, e):
                return True
    return False


def _pressure_finish_block(model, bid, scored_fallback=False):
    """Rank each bay's champion-safe empty-window option by immediate base."""
    TELEMETRY["pressure_finish_calls"] += 1
    np = _np
    blk = model.blocks[bid]
    prefs = blk["bay_preferences"]
    pmax = max(prefs)
    R = _rel_day(blk)
    dur = _dur(blk)
    due = blk["due_date"]
    wl = blk["workload"]
    champion = None
    best = None
    best_base = None

    for b in sorted(range(model.m), key=lambda j: (-prefs[j], j)):
        e = max([R] + [c[5] for c in model.committed.values() if c[0] == b])
        for oi in range(len(blk["shape"])):
            r = model.feasible_map(bid, oi, b, e)
            if r is None:
                continue
            feas, x_lo, y_lo = r
            ys, xs = np.nonzero(feas)
            if len(ys) == 0:
                continue
            option = (oi, b, int(xs[0] + x_lo), int(ys[0] + y_lo), e)
            tardy = max(0.0, float(e + dur) - due)
            base = model.w1 * tardy + model.w3 * (pmax - prefs[b]) \
                + model.w2 * model.z2_delta(b, wl)
            if champion is None:
                champion = option
                best = option
                best_base = base
            elif base < best_base:
                best = option
                best_base = base
            break

    if best is None:
        return False
    oi, b, x, y, e = best
    strict_change = int(best != champion)
    if scored_fallback:
        r = model.feasible_map(bid, oi, b, e)
        if r is not None:
            feas, x_lo, y_lo = r
            ys, xs = np.nonzero(feas)
            if len(ys):
                x, y = _scored_cell(model, bid, oi, b, e, e + dur,
                                    xs, ys, x_lo, y_lo)
    committed = model.commit(bid, oi, b, x, y, e)
    if committed:
        TELEMETRY["ff_obj_strict_changes"] += strict_change
    return committed


def _construct(model, deadline):
    """Greedy construction of all blocks. Under deadline pressure the tail is
    fast-finished (a partial assignment is stage-1 infeasible and worthless).
    Returns True if every block was committed."""
    blocks = model.blocks
    ratio = _constructor_pressure_ratio(model.prob, deadline)
    use_edd = (ratio is not None and
               ratio >= _CONSTRUCTOR_PRESSURE_EDD_THRESHOLD)
    TELEMETRY["constructor_pressure_ratio"] = ratio
    TELEMETRY["constructor_pressure_threshold"] = (
        _CONSTRUCTOR_PRESSURE_EDD_THRESHOLD)
    TELEMETRY["constructor_order"] = "edd" if use_edd else "native"
    TELEMETRY["constructor_route_reason"] = (
        "threshold" if use_edd else
        "feature-fallback" if ratio is None else
        "below-threshold")
    if use_edd:
        order = sorted(
            range(model.n),
            key=lambda bid: (blocks[bid]["due_date"],
                             blocks[bid]["release_time"],
                             -blocks[bid]["workload"], bid))
    else:
        order = sorted(
            range(model.n),
            key=lambda bid: (blocks[bid]["due_date"]
                             - blocks[bid]["release_time"]
                             - blocks[bid]["processing_time"],
                             blocks[bid]["due_date"],
                             -blocks[bid]["workload"]))
    pressure = False
    for bid in order:
        if not pressure and time.monotonic() > deadline:
            pressure = True
        ok = (_pressure_finish_block(model, bid) if pressure
              else _place_block(model, bid))
        if not ok:
            return False
    return True


def _construct_scored(model, soft_deadline, hard_deadline):
    """EDD construction with scored fallback cells and no LNS."""
    blocks = model.blocks
    order = sorted(
        range(model.n),
        key=lambda bid: (blocks[bid]["due_date"],
                         blocks[bid]["release_time"],
                         -blocks[bid]["workload"], bid))
    TELEMETRY["scored_fallback_calls"] = 0
    pressure = False
    try:
        for bid in order:
            now = time.monotonic()
            if now >= hard_deadline:
                return False
            if not pressure and now > soft_deadline:
                pressure = True
            ok = (_pressure_finish_block(model, bid, scored_fallback=True)
                  if pressure else
                  _place_block(model, bid, scored_fallback=True))
            if not ok:
                return False
        return True
    finally:
        TELEMETRY["hedge_bank_scored_calls"] = TELEMETRY.get(
            "scored_fallback_calls", 0)


# ---------------------------------------------------------------------------
# L4 LNS (§5): targeted destroy + regret-2 repair under LAHC acceptance.
# ---------------------------------------------------------------------------

_IT10_FINALISTS_PER_BASE = 8
_IT10_LAMBDA = 0.0
_IT10_PRICE_MIN_TARDY_SHARE = 0.02
_IT11_ZERO_OVERHEAD_LOSS = False
_IT11_LAMBDA_WIN_VALUE = 0.10
_IT11_BUDGET_SKIP_SECONDS = 45.0
_CONSTRUCTOR_PRESSURE_EDD_THRESHOLD = 0.50
_HEDGE_MIN_TIMELIMIT_SECONDS = 45.0
_HEDGE_BANK_MAX_SECONDS = 17.0
_HEDGE_BANK_SOFT_MARGIN_SECONDS = 1.5
_PARALLEL_RACE_MIN_TIMELIMIT_SECONDS = 45.0
_PARALLEL_RACE_PRESSURE_THRESHOLD = 0.60
_PARALLEL_RACE_COMBINED_ESTIMATE_LIMIT = 6_200_000_000
_PARALLEL_RACE_PARENT_RESERVE_SECONDS = 0.40
_HEDGE_ROUTE_OVERRIDE = {"active": False, "ratio": None}
# Disabled by default; IT11_MICRO_THRESHOLD can re-enable a block-count skip.
_IT11_MICRO_BLOCK_THRESHOLD = 0
# Offline z-score moments only (no instance table / nearest-neighbor at runtime).
_IT11_FEATURE_MEANS = {
    "area_pressure_per_day": 0.803011620901322,
    "slack_p10": 0.34000000000000014,
}
_IT11_FEATURE_STDS = {
    "area_pressure_per_day": 0.24642012503376776,
    "slack_p10": 0.4641120554348919,
}
# Pure feature threshold: lambda-win iff z(area_pressure_per_day) in (lo, hi).
# Fit offline from public feature->lambda labels (band recovers 18/20); runtime
# uses only THIS instance's features — no NN, no identity, no anchor table.
_IT11_AREA_Z_WIN_LO = -1.0
_IT11_AREA_Z_WIN_HI = 0.45


def _it10_load_lambda():
    raw = os.environ.get("IT10_LAMBDA", "auto")
    if str(raw).strip().lower() in ("", "auto"):
        return None
    try:
        return max(0.0, float(raw))
    except Exception:
        return None


def _it10_load_price_min_tardy_share():
    try:
        return max(0.0, float(os.environ.get(
            "IT10_PRICE_MIN_TARDY_SHARE", "0.02") or "0.02"))
    except Exception:
        return 0.02


def _it10_tel_inc(key, value=1):
    TELEMETRY[key] = TELEMETRY.get(key, 0) + value


def _it10_incumbent_tardy_immaterial(model):
    obj1 = _INCUMBENT.get("obj1")
    total = _INCUMBENT.get("objective")
    if obj1 is None or total is None:
        return False
    share = (model.w1 * obj1) / max(total, 1)
    return share < _IT10_PRICE_MIN_TARDY_SHARE


def _it10_pricing_material(model):
    return not _it10_incumbent_tardy_immaterial(model)


def _it11_micro_skip(prob_info):
    try:
        threshold = int(os.environ.get(
            "IT11_MICRO_THRESHOLD", str(_IT11_MICRO_BLOCK_THRESHOLD)))
    except Exception:
        threshold = _IT11_MICRO_BLOCK_THRESHOLD
    return len(prob_info.get("blocks", ())) < threshold


def _it11_budget_skip(timelimit):
    try:
        return float(timelimit) < _IT11_BUDGET_SKIP_SECONDS
    except Exception:
        return False


def _it11_mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _it11_percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _it11_polygon_area(points):
    if len(points) < 3:
        return 0.0
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _it11_block_min_area(block):
    orientation_areas = []
    for orientation in block["shape"]:
        layer_areas = []
        for layer in orientation["layers"]:
            layer_areas.append(_it11_polygon_area(layer))
        orientation_areas.append(sum(layer_areas))
    return min(orientation_areas)


def _it11_active_features(prob_info):
    bays = prob_info["bays"]
    blocks = prob_info["blocks"]
    bay_areas = [float(b["width"]) * float(b["height"]) for b in bays]
    total_bay_area = sum(bay_areas)
    releases = [float(b["release_time"]) for b in blocks]
    dues = [float(b["due_date"]) for b in blocks]
    procs = [float(b["processing_time"]) for b in blocks]
    slacks = [d - r - p for r, d, p in zip(releases, dues, procs)]
    total_min_area = sum(_it11_block_min_area(b) for b in blocks)
    horizon_span = max(dues) - min(releases)
    return {
        "area_pressure_per_day":
            total_min_area * _it11_mean(procs) / (total_bay_area * horizon_span),
        "slack_p10": _it11_percentile(slacks, 0.10),
    }


def _it11_block_min_area_bounded(block, deadline):
    orientation_areas = []
    for orientation in block["shape"]:
        _deadline_expired(deadline)
        layer_areas = []
        for layer in orientation["layers"]:
            _deadline_expired(deadline)
            layer_areas.append(_it11_polygon_area(layer))
        orientation_areas.append(sum(layer_areas))
    return min(orientation_areas)


def _constructor_pressure_ratio(prob_info, deadline):
    """Identity-free spatial-temporal offered load; fail closed to None."""
    if _HEDGE_ROUTE_OVERRIDE["active"]:
        return _HEDGE_ROUTE_OVERRIDE["ratio"]
    try:
        _deadline_expired(deadline)
        bays = prob_info["bays"]
        blocks = prob_info["blocks"]
        if not blocks:
            return None
        total_bay_area = sum(float(b["width"]) * float(b["height"])
                             for b in bays)
        releases = [float(block["release_time"]) for block in blocks]
        dues = [float(block["due_date"]) for block in blocks]
        horizon = max(dues) - min(releases)
        if (not math.isfinite(total_bay_area) or total_bay_area <= 0.0 or
                not math.isfinite(horizon) or horizon <= 0.0):
            return None
        area_days = 0.0
        for block in blocks:
            _deadline_expired(deadline)
            proc = float(block["processing_time"])
            area = _it11_block_min_area_bounded(block, deadline)
            if (not math.isfinite(proc) or proc < 0.0 or
                    not math.isfinite(area) or area < 0.0):
                return None
            area_days += area * proc
        ratio = area_days / (total_bay_area * horizon)
        if not math.isfinite(ratio) or ratio < 0.0:
            return None
        return ratio
    except Exception:
        return None


def _it11_active_features_bounded(prob_info, deadline):
    bays = prob_info["bays"]
    blocks = prob_info["blocks"]
    bay_areas = [float(b["width"]) * float(b["height"]) for b in bays]
    total_bay_area = sum(bay_areas)
    releases = [float(b["release_time"]) for b in blocks]
    dues = [float(b["due_date"]) for b in blocks]
    procs = [float(b["processing_time"]) for b in blocks]
    slacks = [d - r - p for r, d, p in zip(releases, dues, procs)]
    total_min_area = 0.0
    for block in blocks:
        _deadline_expired(deadline)
        total_min_area += _it11_block_min_area_bounded(block, deadline)
    _deadline_expired(deadline)
    horizon_span = max(dues) - min(releases)
    return {
        "area_pressure_per_day":
            total_min_area * _it11_mean(procs) / (total_bay_area * horizon_span),
        "slack_p10": _it11_percentile(slacks, 0.10),
    }


def _it11_select_lambda(prob_info):
    """Instance-feature threshold only (no anchors, no NN, no identity)."""
    try:
        feats = _it11_active_features(prob_info)
        area_z = (feats["area_pressure_per_day"]
                  - _IT11_FEATURE_MEANS["area_pressure_per_day"]) \
            / _IT11_FEATURE_STDS["area_pressure_per_day"]
        slack_z = (feats["slack_p10"] - _IT11_FEATURE_MEANS["slack_p10"]) \
            / _IT11_FEATURE_STDS["slack_p10"]
        if _IT11_AREA_Z_WIN_LO < area_z < _IT11_AREA_Z_WIN_HI:
            lambda_used = _IT11_LAMBDA_WIN_VALUE
            path = "lambda-win"
            label = "lambda-win"
        else:
            lambda_used = 0.0
            path = "lambda-loss"
            label = "lambda-loss"
        TELEMETRY["it11_predicted_label"] = label
        TELEMETRY["it11_area_z"] = area_z
        TELEMETRY["it11_slack_z"] = slack_z
        TELEMETRY["it11_lambda_used"] = lambda_used
        TELEMETRY["it11_selector_path"] = path
        return lambda_used
    except Exception:
        TELEMETRY["it11_predicted_label"] = "feature-failure"
        TELEMETRY["it11_area_z"] = None
        TELEMETRY["it11_slack_z"] = None
        TELEMETRY["it11_lambda_used"] = 0.0
        TELEMETRY["it11_selector_path"] = "feature-failure"
        return 0.0


def _it11_select_lambda_bounded(prob_info, deadline):
    """Deadline-aware feature threshold (no anchors, no NN, no identity)."""
    try:
        _deadline_expired(deadline)
        feats = _it11_active_features_bounded(prob_info, deadline)
        area_z = (feats["area_pressure_per_day"]
                  - _IT11_FEATURE_MEANS["area_pressure_per_day"]) \
            / _IT11_FEATURE_STDS["area_pressure_per_day"]
        slack_z = (feats["slack_p10"] - _IT11_FEATURE_MEANS["slack_p10"]) \
            / _IT11_FEATURE_STDS["slack_p10"]
        _deadline_expired(deadline)
        if _IT11_AREA_Z_WIN_LO < area_z < _IT11_AREA_Z_WIN_HI:
            lambda_used = _IT11_LAMBDA_WIN_VALUE
            path = "lambda-win"
            label = "lambda-win"
        else:
            lambda_used = 0.0
            path = "lambda-loss"
            label = "lambda-loss"
        TELEMETRY["it11_predicted_label"] = label
        TELEMETRY["it11_area_z"] = area_z
        TELEMETRY["it11_slack_z"] = slack_z
        TELEMETRY["it11_lambda_used"] = lambda_used
        TELEMETRY["it11_selector_path"] = path
        return lambda_used
    except _SelectorDeadline:
        TELEMETRY["it11_predicted_label"] = "selector-timeout"
        TELEMETRY["it11_area_z"] = None
        TELEMETRY["it11_slack_z"] = None
        TELEMETRY["it11_lambda_used"] = 0.0
        TELEMETRY["it11_selector_path"] = "selector-timeout"
        return 0.0
    except Exception:
        TELEMETRY["it11_predicted_label"] = "feature-failure"
        TELEMETRY["it11_area_z"] = None
        TELEMETRY["it11_slack_z"] = None
        TELEMETRY["it11_lambda_used"] = 0.0
        TELEMETRY["it11_selector_path"] = "feature-failure"
        return 0.0


def _armor_l2_estimate(prob_info):
    bays = prob_info["bays"]
    blocks = prob_info["blocks"]
    area_cells = sum(int(b["width"]) * int(b["height"]) for b in bays)
    K = max(1, max((len(s["layers"]) for blk in blocks
                    for s in blk["shape"]), default=1))
    n = len(blocks)
    resident_days_bound = sum(_dur(blk) for blk in blocks)
    event_days_bound = 2 * n
    raw_grid_bytes = 2 * area_cells * K * (
        2 * resident_days_bound + event_days_bound)
    estimated_peak_bytes = _ARMOR_L2_BASE_BYTES + (5 * raw_grid_bytes) // 2
    return {
        "area_cells": area_cells,
        "K": K,
        "n": n,
        "resident_days_bound": resident_days_bound,
        "event_days_bound": event_days_bound,
        "raw_grid_bytes": raw_grid_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
    }


def _it10_edge_gap_bucket(gap):
    if gap <= 0:
        return "0"
    if gap == 1:
        return "1"
    if gap <= 3:
        return "2-3"
    if gap <= 7:
        return "4-7"
    if gap <= 15:
        return "8-15"
    return "16+"


def _it10_record_shadow(stats):
    if stats is None:
        return
    gap = stats["edge_gap"]
    _it10_tel_inc("it10_shadow_placements")
    _it10_tel_inc("it10_shadow_selected_edges_sum",
                  stats["chosen_edges"])
    _it10_tel_inc("it10_shadow_alt_min_edges_sum",
                  stats["min_edges_same_base_bucket"])
    _it10_tel_inc("it10_shadow_edge_gap_sum", gap)
    if gap > 0:
        _it10_tel_inc("it10_shadow_edge_gap_positive")
    _it10_tel_inc("it10_shadow_chosen_price_sum",
                  stats["chosen_shadow_price"])
    _it10_tel_inc("it10_shadow_eval_ms_sum", stats["shadow_eval_ms"])
    _it10_tel_inc("it10_shadow_eval_candidates",
                  stats["shadow_eval_candidates"])
    _it10_tel_inc("it10_shadow_exact_edges_sum",
                  stats["shadow_exact_edges_sum"])
    hist = TELEMETRY.setdefault("it10_shadow_edge_gap_hist", {})
    bucket = _it10_edge_gap_bucket(gap)
    hist[bucket] = hist.get(bucket, 0) + 1


class _RepairRiskIndex:
    """Lambda-0 IT10 probe index: future event opportunities by bay/day."""

    __slots__ = ("model", "records", "by_owner")

    def __init__(self, model, remaining, saved):
        self.model = model
        self.records = {}
        self.by_owner = {}
        remaining = set(remaining)
        if saved:
            for bid in remaining:
                c = saved.get(bid)
                if c is not None:
                    self._add_saved_window(bid, c)
        self._add_tardy_watchlist(max(4, len(remaining)), remaining)
        _it10_tel_inc("it10_risk_index_builds")
        _it10_tel_inc("it10_risk_index_records", self.record_count())

    def record_count(self):
        return sum(len(v) for v in self.records.values())

    def _add_event(self, bid, placement, event_day, kind, candidate_entry):
        b, oi, x, y = placement[:4]
        if event_day < 0:
            return
        key = (bid, kind, event_day, candidate_entry)
        owner_key = (bid, oi, x, y)
        rec = (bid, kind, event_day, candidate_entry, owner_key)
        bucket = (b, event_day)
        self.records.setdefault(bucket, []).append(rec)
        self.by_owner.setdefault(bid, []).append((bucket, key, rec))

    def _add_candidate_days(self, bid, c, days):
        blk = self.model.blocks[bid]
        dur = _dur(blk)
        rel = _rel_day(blk)
        seen = set()
        for ce in days:
            if ce < rel or ce in seen:
                continue
            seen.add(ce)
            self._add_event(bid, c, ce, "entry", ce)
            self._add_event(bid, c, ce + dur, "exit", ce)

    def _add_saved_window(self, bid, c):
        e = c[4]
        self._add_candidate_days(bid, c, (e - 2, e - 1, e, e + 1, e + 2))

    def _add_tardy_watchlist(self, limit, remaining):
        watch = []
        for bid, c in self.model.committed.items():
            if bid in remaining:
                continue
            tardy = max(0.0, float(c[5]) - self.model.blocks[bid]["due_date"])
            if tardy > 0:
                watch.append((tardy, bid, c))
        watch.sort(reverse=True)
        for _, bid, c in watch[:limit]:
            e = c[4]
            self._add_candidate_days(bid, c, (e - 2, e - 1, e))

    def remove_owner(self, bid):
        items = self.by_owner.pop(bid, None)
        if not items:
            return
        for bucket, key, rec in items:
            lst = self.records.get(bucket)
            if not lst:
                continue
            self.records[bucket] = [r for r in lst
                                    if (r[0], r[1], r[2], r[3]) != key]
            if not self.records[bucket]:
                del self.records[bucket]

    def price(self, bid, oi, b, x, y, e, xd):
        cand_key = (bid, oi, x, y)
        seen = set()
        edges = 0
        for t in range(e + 1, xd):
            for owner, kind, day, candidate_entry, owner_key in \
                    self.records.get((b, t), ()):
                if owner == bid:
                    continue
                edge_key = (owner, kind, day, candidate_entry)
                if edge_key in seen:
                    continue
                if self.model._obstructs(cand_key, owner_key):
                    seen.add(edge_key)
                    edges += 1
        return edges, edges * self.model.w1


def _it10_shadow_for_choice(model, bid, placement, chosen_base, risk_index):
    np = _np
    blk = model.blocks[bid]
    prefs = blk["bay_preferences"]
    pmax = max(prefs)
    R = _rel_day(blk)
    dur = _dur(blk)
    due = blk["due_date"]
    wl = blk["workload"]
    nor = len(blk["shape"])
    shadow_min_edges = None
    shadow_eval_ms = 0.0
    shadow_eval_candidates = 0
    shadow_exact_edges_sum = 0

    for e in (R, R + 1, R + 2):
        tardy = max(0.0, float(e + dur) - due)
        for b in range(model.m):
            base = model.w1 * tardy + model.w3 * (pmax - prefs[b]) \
                + model.w2 * model.z2_delta(b, wl)
            if base != chosen_base:
                continue
            xd = e + dur
            for oi in range(nor):
                r = model.feasible_map(bid, oi, b, e)
                if r is None:
                    continue
                feas, x_lo, y_lo = r
                ys, xs = np.nonzero(feas)
                if len(ys) == 0:
                    continue
                om = model.om(bid, oi)
                conv = model.contact_map(bid, oi, b, e, xd)
                W, H = model.bays[b]
                axs = xs + x_lo
                ays = ys + y_lo
                contact = model.contact_at(conv, om, axs, ays)
                cxs = axs + (om.bbox[0] + om.bbox[2]) * 0.5
                cys = ays + (om.bbox[1] + om.bbox[3]) * 0.5
                d2 = (cxs - W * 0.5) ** 2 + (cys - H * 0.5) ** 2
                cell = contact + 0.5 * d2 / (W * W + H * H)
                i = int(np.argmax(cell))
                k = min(_IT10_FINALISTS_PER_BASE, len(cell))
                if k == len(cell):
                    idxs = np.arange(len(cell))
                else:
                    idxs = np.argpartition(cell, len(cell) - k)[-k:]
                if not (idxs == i).any():
                    idxs = np.append(idxs, i)
                for ii in idxs:
                    t0 = time.monotonic()
                    edges, _ = risk_index.price(
                        bid, oi, b, int(axs[ii]), int(ays[ii]), e, xd)
                    shadow_eval_ms += (time.monotonic() - t0) * 1000.0
                    shadow_eval_candidates += 1
                    shadow_exact_edges_sum += edges
                    if shadow_min_edges is None or edges < shadow_min_edges:
                        shadow_min_edges = edges

    oi, b, x, y, e = placement
    xd = e + dur
    t0 = time.monotonic()
    chosen_edges, chosen_shadow_price = risk_index.price(
        bid, oi, b, x, y, e, xd)
    shadow_eval_ms += (time.monotonic() - t0) * 1000.0
    shadow_eval_candidates += 1
    shadow_exact_edges_sum += chosen_edges
    if shadow_min_edges is None:
        shadow_min_edges = chosen_edges
    return {
        "chosen_edges": chosen_edges,
        "min_edges_same_base_bucket": shadow_min_edges,
        "edge_gap": chosen_edges - shadow_min_edges,
        "chosen_shadow_price": chosen_shadow_price,
        "shadow_eval_ms": shadow_eval_ms,
        "shadow_eval_candidates": shadow_eval_candidates,
        "shadow_exact_edges_sum": shadow_exact_edges_sum,
    }


def _enum_for_repair(model, bid, risk_index=None, price_enabled=True):
    """Best placement across windows R..R+2 x all bays x all orients, plus an
    optimistic second-best score for regret-2. Returns
    (placement, s1, s2, price_influenced, unpriced_base) where placement =
    (oi, b, x, y, e), or None if no feasible candidate in the normal windows
    (=> most-constrained)."""
    np = _np
    blk = model.blocks[bid]
    prefs = blk["bay_preferences"]
    pmax = max(prefs)
    R = _rel_day(blk)
    dur = _dur(blk)
    due = blk["due_date"]
    wl = blk["workload"]
    nor = len(blk["shape"])
    combos = []
    for e in (R, R + 1, R + 2):
        tardy = max(0.0, float(e + dur) - due)
        for b in range(model.m):
            base = model.w1 * tardy + model.w3 * (pmax - prefs[b]) \
                + model.w2 * model.z2_delta(b, wl)
            combos.append((base, e, b))
    combos.sort()
    if price_enabled and _IT10_LAMBDA > 0.0 and risk_index is not None:
        return _enum_for_repair_priced(model, bid, combos, risk_index,
                                       dur, nor)

    best = None
    s1 = None
    s2 = None
    for base, e, b in combos:
        if s1 is not None and base > s1:
            s2 = base
            break
        xd = e + dur
        for oi in range(nor):
            r = model.feasible_map(bid, oi, b, e)
            if r is None:
                continue
            feas, x_lo, y_lo = r
            ys, xs = np.nonzero(feas)
            if len(ys) == 0:
                continue
            om = model.om(bid, oi)
            conv = model.contact_map(bid, oi, b, e, xd)
            W, H = model.bays[b]
            axs = xs + x_lo
            ays = ys + y_lo
            contact = model.contact_at(conv, om, axs, ays)
            cxs = axs + (om.bbox[0] + om.bbox[2]) * 0.5
            cys = ays + (om.bbox[1] + om.bbox[3]) * 0.5
            d2 = (cxs - W * 0.5) ** 2 + (cys - H * 0.5) ** 2
            cell = contact + 0.5 * d2 / (W * W + H * H)
            i = int(np.argmax(cell))
            cand = (base, -float(cell[i]), oi, b, int(axs[i]), int(ays[i]), e)
            if best is None or cand < best:
                best = cand
                s1 = base
    if best is None:
        return None
    if s2 is None:
        s2 = s1  # only one feasible base level seen: zero regret
    return (best[2], best[3], best[4], best[5], best[6]), s1, s2, False, s1


def _enum_for_repair_priced(model, bid, combos, risk_index, dur, nor):
    np = _np
    best = None
    s1 = None
    s2 = None
    unpriced_best = None
    chosen_base = None

    for base, e, b in combos:
        if s1 is not None and base > s1:
            if s2 is None or base < s2:
                s2 = base
            break
        xd = e + dur
        for oi in range(nor):
            r = model.feasible_map(bid, oi, b, e)
            if r is None:
                continue
            feas, x_lo, y_lo = r
            ys, xs = np.nonzero(feas)
            if len(ys) == 0:
                continue
            om = model.om(bid, oi)
            conv = model.contact_map(bid, oi, b, e, xd)
            W, H = model.bays[b]
            axs = xs + x_lo
            ays = ys + y_lo
            contact = model.contact_at(conv, om, axs, ays)
            cxs = axs + (om.bbox[0] + om.bbox[2]) * 0.5
            cys = ays + (om.bbox[1] + om.bbox[3]) * 0.5
            d2 = (cxs - W * 0.5) ** 2 + (cys - H * 0.5) ** 2
            cell = contact + 0.5 * d2 / (W * W + H * H)
            i = int(np.argmax(cell))
            unpriced = (base, -float(cell[i]), oi, b, int(axs[i]),
                        int(ays[i]), e)
            if unpriced_best is None or unpriced < unpriced_best:
                unpriced_best = unpriced

            k = min(_IT10_FINALISTS_PER_BASE, len(cell))
            if k == len(cell):
                idxs = np.arange(len(cell))
            else:
                idxs = np.argpartition(cell, len(cell) - k)[-k:]
            if not (idxs == i).any():
                idxs = np.append(idxs, i)
            for ii in idxs:
                _, raw_price = risk_index.price(
                    bid, oi, b, int(axs[ii]), int(ays[ii]), e, xd)
                total = base + _IT10_LAMBDA * raw_price
                cand = (total, -float(cell[ii]), oi, b, int(axs[ii]),
                        int(ays[ii]), e, base)
                if best is None or cand < best:
                    if best is not None:
                        if s2 is None or best[0] < s2:
                            s2 = best[0]
                    best = cand
                    s1 = total
                    chosen_base = base
                else:
                    if s2 is None or total < s2:
                        s2 = total

    if best is None:
        return None
    if s2 is None:
        s2 = s1
    placement = (best[2], best[3], best[4], best[5], best[6])
    unpriced_placement = (unpriced_best[2], unpriced_best[3],
                          unpriced_best[4], unpriced_best[5],
                          unpriced_best[6])
    influenced = placement != unpriced_placement
    return placement, s1, s2, influenced, chosen_base


def _enum_for_repair_champion(model, bid):
    """Champion repair enumerator kept separate for the selector loss path."""
    np = _np
    blk = model.blocks[bid]
    prefs = blk["bay_preferences"]
    pmax = max(prefs)
    R = _rel_day(blk)
    dur = _dur(blk)
    due = blk["due_date"]
    wl = blk["workload"]
    nor = len(blk["shape"])
    combos = []
    for e in (R, R + 1, R + 2):
        tardy = max(0.0, float(e + dur) - due)
        for b in range(model.m):
            base = model.w1 * tardy + model.w3 * (pmax - prefs[b]) \
                + model.w2 * model.z2_delta(b, wl)
            combos.append((base, e, b))
    combos.sort()
    best = None
    s1 = None
    s2 = None
    for base, e, b in combos:
        if s1 is not None and base > s1:
            s2 = base
            break
        xd = e + dur
        for oi in range(nor):
            r = model.feasible_map(bid, oi, b, e)
            if r is None:
                continue
            feas, x_lo, y_lo = r
            ys, xs = np.nonzero(feas)
            if len(ys) == 0:
                continue
            om = model.om(bid, oi)
            conv = model.contact_map(bid, oi, b, e, xd)
            W, H = model.bays[b]
            axs = xs + x_lo
            ays = ys + y_lo
            contact = model.contact_at(conv, om, axs, ays)
            cxs = axs + (om.bbox[0] + om.bbox[2]) * 0.5
            cys = ays + (om.bbox[1] + om.bbox[3]) * 0.5
            d2 = (cxs - W * 0.5) ** 2 + (cys - H * 0.5) ** 2
            cell = contact + 0.5 * d2 / (W * W + H * H)
            i = int(np.argmax(cell))
            cand = (base, -float(cell[i]), oi, b, int(axs[i]), int(ays[i]), e)
            if best is None or cand < best:
                best = cand
                s1 = base
    if best is None:
        return None
    if s2 is None:
        s2 = s1
    return (best[2], best[3], best[4], best[5], best[6]), s1, s2


def _repair_champion(model, victims, deadline):
    """Champion repair loop for selector loss/fallback paths."""
    remaining = list(victims)
    cands = {}
    while remaining:
        if time.monotonic() > deadline:
            for bid in remaining:
                if time.monotonic() > deadline:
                    return False
                if not (_fast_finish_block(model, bid)
                        or _place_block(model, bid)):
                    return False
            return True
        chosen = None
        for bid in remaining:
            if bid not in cands:
                cands[bid] = _enum_for_repair_champion(model, bid)
            c = cands[bid]
            if c is None:
                chosen = (float("inf"), 0.0, bid)
                break
            regret = c[2] - c[1]
            key = (regret, c[1], bid)
            if chosen is None or key > chosen:
                chosen = key
        bid = chosen[-1]
        c = cands.pop(bid, None)
        ok = False
        ins_bay = None
        if c is not None:
            oi, b, x, y, e = c[0]
            ok = model.commit(bid, oi, b, x, y, e)
            ins_bay = b
        if not ok:
            ok = _place_block(model, bid)
            if ok:
                ins_bay = model.committed[bid][0]
        if not ok:
            return False
        remaining.remove(bid)
        for other in list(cands):
            cc = cands[other]
            if cc is None or cc[0][1] == ins_bay:
                del cands[other]
    return True


def _repair(model, victims, deadline, saved=None):
    """Regret-2 reinsertion. Returns False only if some block cannot be
    placed at all (lost run; caller restores)."""
    remaining = list(victims)
    risk_index = None
    price_enabled = False
    if not _IT11_ZERO_OVERHEAD_LOSS:
        risk_index = _RepairRiskIndex(model, remaining, saved)
        price_enabled = True
    if price_enabled and _IT10_LAMBDA > 0.0 and not _it10_pricing_material(model):
        price_enabled = False
        _it10_tel_inc("it10_price_skipped_no_tardy")
    cands = {}
    while remaining:
        if time.monotonic() > deadline:
            for bid in remaining:
                if time.monotonic() > deadline:
                    return False
                if not (_fast_finish_block(model, bid)
                        or _place_block(model, bid)):
                    return False
            return True
        chosen = None
        for bid in remaining:
            if bid not in cands:
                cands[bid] = _enum_for_repair(model, bid, risk_index,
                                              price_enabled)
            c = cands[bid]
            if c is None:
                chosen = (float("inf"), 0.0, bid)   # most constrained first
                break
            regret = c[2] - c[1]
            key = (regret, c[1], bid)
            if chosen is None or key > chosen:
                chosen = key
        bid = chosen[-1]
        c = cands.pop(bid, None)
        ok = False
        ins_bay = None
        if c is not None:
            oi, b, x, y, e = c[0]
            shadow = None
            if risk_index is not None:
                shadow_base = c[4] if len(c) > 4 else c[1]
                shadow = _it10_shadow_for_choice(model, bid, c[0],
                                                 shadow_base, risk_index)
            ok = model.commit(bid, oi, b, x, y, e)
            ins_bay = b
            if ok and risk_index is not None:
                if _IT10_LAMBDA > 0.0 and price_enabled:
                    _it10_tel_inc("it10_priced_placements")
                    if len(c) > 3 and c[3]:
                        _it10_tel_inc("it10_price_influenced")
                _it10_record_shadow(shadow)
        if not ok:
            ok = _place_block(model, bid)
            if ok:
                ins_bay = model.committed[bid][0]
        if not ok:
            return False
        remaining.remove(bid)
        if risk_index is not None:
            risk_index.remove_owner(bid)
        # bay-local grids changed: candidate caches in that bay are stale
        for other in list(cands):
            cc = cands[other]
            if cc is None or cc[0][1] == ins_bay:
                del cands[other]
    return True


def _destroy_random(model, rng, s):
    bids = list(model.committed)
    rng.shuffle(bids)
    return bids[:s]


def _destroy_worst_tardy(model, rng, s):
    tardy = [(max(0.0, c[5] - model.blocks[bid]["due_date"]), rng.random(), bid)
             for bid, c in model.committed.items()]
    tardy = [t for t in tardy if t[0] > 0]
    if not tardy:
        return _destroy_random(model, rng, s)
    tardy.sort(reverse=True)
    out = [t[2] for t in tardy[:s]]
    if len(out) < s:
        pool = [b for b in model.committed if b not in set(out)]
        rng.shuffle(pool)
        out += pool[:s - len(out)]
    return out


def _destroy_time_band(model, rng, s):
    if not model.committed:
        return []
    b = rng.randrange(model.m)
    in_bay = [(bid, c) for bid, c in model.committed.items() if c[0] == b]
    if not in_bay:
        return _destroy_random(model, rng, s)
    _, c0 = in_bay[rng.randrange(len(in_bay))]
    w = rng.randint(3, 7)
    d0 = c0[4]
    out = [bid for bid, c in in_bay if c[4] < d0 + w and d0 < c[5]]
    rng.shuffle(out)
    return out[:max(s, 12) + 4]


def _destroy_pref_regret(model, rng, s):
    reg = []
    for bid, c in model.committed.items():
        prefs = model.blocks[bid]["bay_preferences"]
        r = max(prefs) - prefs[c[0]]
        if r > 0:
            reg.append((r, rng.random(), bid))
    if not reg:
        return _destroy_random(model, rng, s)
    reg.sort(reverse=True)
    return [t[2] for t in reg[:s]]


_DESTROY_OPS = (_destroy_random, _destroy_worst_tardy,
                _destroy_time_band, _destroy_pref_regret)


def _snapshot(model):
    return {bid: c[:5] for bid, c in model.committed.items()}


def _restore(model, snap):
    for bid in list(model.committed):
        model.uncommit(bid)
    for bid, (b, oi, x, y, e) in snap.items():
        if not model.commit(bid, oi, b, x, y, e):
            raise RuntimeError(f"restore failed for block {bid}")


def _lns_run(model, prob_info, t0, timelimit, base_reserve, check_cost):
    """Improve in place until the deadline; promote utils-validated bests
    into _INCUMBENT. Never raises on its own account."""
    rng = random.Random(int(os.environ.get("OGC_SEED", "20260615")))
    dmax = 8 if timelimit < 45 else 12
    cur = model.objective()[0]
    best = cur
    best_snap = _snapshot(model)
    hist = [cur] * 50
    it = accepts = promotions = rejects_in_row = 0
    last_promote_t = time.monotonic()
    promoted_obj = _INCUMBENT["objective"]
    loss_path = _IT11_ZERO_OVERHEAD_LOSS

    def deadline():
        return t0 + timelimit - (base_reserve + 3 * check_cost[0])

    def promote_current(obj_now):
        nonlocal promotions
        sol = model.emit()
        tc = time.monotonic()
        res = _validate(prob_info, sol)
        # latest measurement, not running max: one transient stall must not
        # permanently shrink the search window (§6 "re-measured")
        check_cost[0] = time.monotonic() - tc
        if res is not None and res.get("feasible"):
            if obj_now != res["objective"]:
                TELEMETRY["objective_drift"] = (obj_now, res["objective"])
            if _INCUMBENT["objective"] is None or \
                    res["objective"] < _INCUMBENT["objective"]:
                _INCUMBENT.update(solution=sol, objective=res["objective"],
                                  obj1=res.get("obj1"))
                promotions += 1
                return res["objective"]
        else:
            TELEMETRY["promotion_rejected"] = \
                TELEMETRY.get("promotion_rejected", 0) + 1
        return None

    while True:
        now = time.monotonic()
        if _STOP["flag"] or now > deadline():
            break
        s = rng.randint(4, dmax)
        op = _DESTROY_OPS[rng.randrange(len(_DESTROY_OPS))]
        victims = op(model, rng, s)
        if not victims:
            break
        saved = {bid: model.committed[bid][:5] for bid in victims}
        for bid in victims:
            model.uncommit(bid)
        if loss_path:
            repaired = _repair_champion(model, set(saved), deadline())
        else:
            repaired = _repair(model, set(saved), deadline(), saved)
        if not repaired:
            for bid in list(saved):
                if bid in model.committed:
                    model.uncommit(bid)
            for bid, (b, oi, x, y, e) in saved.items():
                model.commit(bid, oi, b, x, y, e)
            it += 1
            continue
        cand = model.objective()[0]
        if cand <= hist[it % 50] or cand <= cur:
            cur = cand
            accepts += 1
            rejects_in_row = 0
            if cur < best - 1e-9:
                best = cur
                best_snap = _snapshot(model)
                now = time.monotonic()
                if promoted_obj is None or \
                        best < promoted_obj * 0.995 or \
                        now - last_promote_t > 2.0:
                    got = promote_current(best)
                    if got is not None:
                        promoted_obj = got
                    last_promote_t = now
        else:
            for bid in list(saved):
                model.uncommit(bid)
            for bid, (b, oi, x, y, e) in saved.items():
                if not model.commit(bid, oi, b, x, y, e):
                    raise RuntimeError(f"revert failed for block {bid}")
            rejects_in_row += 1
        hist[it % 50] = cur
        it += 1
        if rejects_in_row >= 150 and time.monotonic() < deadline() - 0.25:
            _restore(model, best_snap)
            kick = min(len(model.committed), max(dmax * 3, 20))
            victims = _destroy_random(model, rng, kick) if kick else []
            saved = {bid: model.committed[bid][:5] for bid in victims}
            for bid in victims:
                model.uncommit(bid)
            if loss_path:
                repaired = bool(victims) and \
                    _repair_champion(model, set(saved), deadline())
            else:
                repaired = bool(victims) and \
                    _repair(model, set(saved), deadline(), saved)
            if repaired:
                cur = model.objective()[0]
            else:
                _restore(model, best_snap)
                cur = best
            hist = [cur] * 50
            rejects_in_row = 0

    # final: make sure the internal best was offered to the incumbent
    if promoted_obj is None or best < promoted_obj - 1e-9:
        if cur > best:
            _restore(model, best_snap)
        if time.monotonic() + check_cost[0] < t0 + timelimit \
                - (base_reserve * 0.5):
            promote_current(best)
    TELEMETRY["lns_iterations"] = it
    TELEMETRY["lns_accepts"] = accepts
    TELEMETRY["lns_promotions"] = promotions
    TELEMETRY["lns_best_internal"] = best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _best_solution():
    sol = _INCUMBENT["solution"]
    if sol is not None:
        return sol
    lr = _INCUMBENT["last_resort"]
    return lr if lr is not None else {"operations": {}}


def _legacy_algorithm(prob_info, timelimit=60):
    global _IT10_LAMBDA, _IT10_PRICE_MIN_TARDY_SHARE
    global _IT11_ZERO_OVERHEAD_LOSS
    t0 = time.monotonic()
    _INCUMBENT.update(solution=None, objective=None, obj1=None,
                      last_resort=None)
    _STOP.update(flag=False, final=False)
    TELEMETRY.clear()
    _reset_kernel_telemetry("public/legacy")
    _IT11_ZERO_OVERHEAD_LOSS = False
    if _it11_budget_skip(timelimit):
        _IT10_LAMBDA = 0.0
        _IT11_ZERO_OVERHEAD_LOSS = True
        TELEMETRY["it11_selector_path"] = "budget-skip"
        t0 = time.monotonic()
    elif _it11_micro_skip(prob_info):
        _IT10_LAMBDA = 0.0
        _IT11_ZERO_OVERHEAD_LOSS = True
        TELEMETRY["it11_selector_path"] = "micro-skip"
        t0 = time.monotonic()
    else:
        lambda_override = _it10_load_lambda()
        if lambda_override is None:
            _IT10_LAMBDA = _it11_select_lambda(prob_info)
            _IT11_ZERO_OVERHEAD_LOSS = (
                TELEMETRY.get("it11_selector_path") != "lambda-win")
        else:
            _IT10_LAMBDA = lambda_override
            TELEMETRY["it11_predicted_label"] = "override"
            TELEMETRY["it11_area_z"] = None
            TELEMETRY["it11_lambda_used"] = _IT10_LAMBDA
            TELEMETRY["it11_selector_path"] = "override"
    if _IT11_ZERO_OVERHEAD_LOSS:
        t0 = time.monotonic()
    else:
        _IT10_PRICE_MIN_TARDY_SHARE = _it10_load_price_min_tardy_share()
        TELEMETRY["it10_lambda"] = _IT10_LAMBDA
    prev_handler = None
    armed = False
    try:
        _cap_thread_pools()

        gap = _jitter_probe(0.1)
        reserve_jitter = min(2.0, 10.0 * gap)
        check_cost = 0.25  # initial guess; re-measured at first validation
        reserve = max(2.0, 0.04 * timelimit) + reserve_jitter + 3 * check_cost

        # SIGALRM backstop (§6): main thread only; handler raises a private
        # exception unless the return path has already begun.
        if threading.current_thread() is threading.main_thread():
            try:
                def _on_alarm(signum, frame):
                    _STOP["flag"] = True
                    if not _STOP["final"]:
                        raise _DeadlineAlarm()
                prev_handler = signal.signal(signal.SIGALRM, _on_alarm)
                signal.alarm(max(1, math.ceil(timelimit - reserve / 2.0)))
                armed = True
            except _DeadlineAlarm:
                raise
            except Exception:
                prev_handler = None

        # L1: validated incumbent within ~0.2 s.
        sol = _l1_sparse(prob_info)
        _INCUMBENT["last_resort"] = sol
        tc = time.monotonic()
        res = _validate(prob_info, sol)
        check_cost = time.monotonic() - tc
        if res is not None and res.get("feasible"):
            _INCUMBENT.update(solution=sol, objective=res["objective"],
                              obj1=res.get("obj1"))
            TELEMETRY["ttfi"] = time.monotonic() - t0
            TELEMETRY["l1_objective"] = res["objective"]
        reserve = max(2.0, 0.04 * timelimit) + reserve_jitter + 3 * check_cost
        TELEMETRY["reserve"] = reserve
        TELEMETRY["jitter_gap"] = gap

        # L2 model + L3 constructor.
        base_reserve = max(2.0, 0.04 * timelimit) + reserve_jitter
        deadline = t0 + timelimit - reserve
        model = _Model(prob_info)
        tcon = time.monotonic()
        complete = _construct(model, deadline - 2 * check_cost - 0.5)
        TELEMETRY["construct_s"] = time.monotonic() - tcon
        constructed_ok = False
        if complete and time.monotonic() < deadline - 1.5 * check_cost:
            sol = model.emit()
            tc = time.monotonic()
            res = _validate(prob_info, sol)
            check_cost = time.monotonic() - tc
            if res is not None and res.get("feasible"):
                constructed_ok = True
                obj_int = model.objective()[0]
                if obj_int != res["objective"]:
                    TELEMETRY["objective_drift"] = (obj_int, res["objective"])
                if _INCUMBENT["objective"] is None or \
                        res["objective"] < _INCUMBENT["objective"]:
                    _INCUMBENT.update(solution=sol,
                                      objective=res["objective"],
                                      obj1=res.get("obj1"))
                    TELEMETRY["construct_objective"] = res["objective"]
                    TELEMETRY["ttci"] = time.monotonic() - t0

        # L4 LNS until deadline.
        if constructed_ok and not _STOP["flag"] and \
                os.environ.get("OGC_DISABLE_LNS") != "1":
            _lns_run(model, prob_info, t0, timelimit, base_reserve,
                     [check_cost])

        _STOP["final"] = True
        TELEMETRY["wall"] = time.monotonic() - t0
        return _best_solution()
    except Exception:
        _STOP["final"] = True
        return _best_solution()
    finally:
        if armed:
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev_handler)
            except Exception:
                pass


def _submission7_armored_algorithm(prob_info, timelimit=60):
    global _IT10_LAMBDA, _IT10_PRICE_MIN_TARDY_SHARE
    global _IT11_ZERO_OVERHEAD_LOSS
    t0 = time.monotonic()
    _INCUMBENT.update(solution=None, objective=None, obj1=None,
                      last_resort=None)
    _STOP.update(flag=False, final=False)
    TELEMETRY.clear()
    _reset_kernel_telemetry("unknown/armored")
    TELEMETRY["armor_path"] = "hidden"
    _IT10_LAMBDA = 0.0
    _IT10_PRICE_MIN_TARDY_SHARE = 0.02
    _IT11_ZERO_OVERHEAD_LOSS = True
    allow_env = _hidden_env_tuning_allowed()
    TELEMETRY["env_tuning_allowed"] = allow_env
    _mark_validator_available()

    prev_handler = None
    armed = False
    try:
        timelimit_f = float(timelimit)
        initial_reserve = max(2.0, 0.04 * timelimit_f) + 0.75
        if threading.current_thread() is threading.main_thread():
            try:
                def _on_alarm(signum, frame):
                    _STOP["flag"] = True
                    if not _STOP["final"]:
                        raise _DeadlineAlarm()
                prev_handler = signal.signal(signal.SIGALRM, _on_alarm)
                signal.alarm(max(1, math.ceil(
                    timelimit_f - initial_reserve / 2.0)))
                armed = True
            except _DeadlineAlarm:
                raise
            except Exception:
                prev_handler = None

        _cap_thread_pools()

        selector_budget = min(0.50, max(0.05, 0.01 * timelimit_f))
        selector_deadline = min(t0 + selector_budget,
                                t0 + timelimit_f - initial_reserve)
        lambda_override = _it10_load_lambda() if allow_env else None
        if lambda_override is None:
            _IT10_LAMBDA = _it11_select_lambda_bounded(
                prob_info, selector_deadline)
            _IT11_ZERO_OVERHEAD_LOSS = (
                TELEMETRY.get("it11_selector_path") != "lambda-win")
        else:
            _IT10_LAMBDA = lambda_override
            _IT11_ZERO_OVERHEAD_LOSS = (_IT10_LAMBDA <= 0.0)
            TELEMETRY["it11_predicted_label"] = "override"
            TELEMETRY["it11_area_z"] = None
            TELEMETRY["it11_lambda_used"] = _IT10_LAMBDA
            TELEMETRY["it11_selector_path"] = "override"
        if _IT11_ZERO_OVERHEAD_LOSS:
            _IT10_LAMBDA = 0.0
        else:
            _IT10_PRICE_MIN_TARDY_SHARE = (
                _it10_load_price_min_tardy_share() if allow_env else 0.02)
            TELEMETRY["it10_lambda"] = _IT10_LAMBDA

        gap = _jitter_probe(0.1)
        reserve_jitter = min(2.0, 10.0 * gap)
        check_cost = 0.25
        reserve = max(2.0, 0.04 * timelimit_f) + reserve_jitter \
            + 3 * check_cost

        sol = _l1_sparse(prob_info)
        _INCUMBENT["last_resort"] = sol
        if time.monotonic() < t0 + timelimit_f - 0.05:
            tc = time.monotonic()
            res = _validate(prob_info, sol)
            check_cost = time.monotonic() - tc
            if res is not None and res.get("feasible"):
                _INCUMBENT.update(solution=sol, objective=res["objective"],
                                  obj1=res.get("obj1"))
                TELEMETRY["ttfi"] = time.monotonic() - t0
                TELEMETRY["l1_objective"] = res["objective"]
            elif TELEMETRY.get("validator_available") is not False:
                TELEMETRY["validator_available"] = False
        reserve = max(2.0, 0.04 * timelimit_f) + reserve_jitter \
            + 3 * check_cost
        TELEMETRY["reserve"] = reserve
        TELEMETRY["jitter_gap"] = gap

        estimate = _armor_l2_estimate(prob_info)
        TELEMETRY["armor_l2_area_cells"] = estimate["area_cells"]
        TELEMETRY["armor_l2_K"] = estimate["K"]
        TELEMETRY["armor_l2_resident_days_bound"] = \
            estimate["resident_days_bound"]
        TELEMETRY["armor_l2_raw_grid_bytes"] = estimate["raw_grid_bytes"]
        TELEMETRY["armor_l2_estimated_peak_bytes"] = \
            estimate["estimated_peak_bytes"]
        TELEMETRY["armor_model_constructed"] = False
        if estimate["estimated_peak_bytes"] > _ARMOR_L2_RSS_CAP_BYTES:
            TELEMETRY["armor_l2_memory_skip"] = True
            TELEMETRY["armor_l2_rss_mb"] = _rss_mb()
            _cheap_l1_improvement(prob_info, t0, timelimit_f, check_cost)
            _STOP["final"] = True
            TELEMETRY["wall"] = time.monotonic() - t0
            return _best_solution()
        TELEMETRY["armor_l2_memory_skip"] = False

        base_reserve = max(2.0, 0.04 * timelimit_f) + reserve_jitter
        deadline = t0 + timelimit_f - reserve
        if time.monotonic() >= deadline or _STOP["flag"]:
            _STOP["final"] = True
            TELEMETRY["wall"] = time.monotonic() - t0
            return _best_solution()

        TELEMETRY["armor_model_constructed"] = True
        model = _Model(prob_info)
        tcon = time.monotonic()
        complete = _construct(model, deadline - 2 * check_cost - 0.5)
        TELEMETRY["construct_s"] = time.monotonic() - tcon
        constructed_ok = False
        if complete and time.monotonic() < deadline - 1.5 * check_cost:
            sol = model.emit()
            tc = time.monotonic()
            res = _validate(prob_info, sol)
            check_cost = time.monotonic() - tc
            if res is not None and res.get("feasible"):
                constructed_ok = True
                obj_int = model.objective()[0]
                if obj_int != res["objective"]:
                    TELEMETRY["objective_drift"] = (obj_int, res["objective"])
                if _INCUMBENT["objective"] is None or \
                        res["objective"] < _INCUMBENT["objective"]:
                    _INCUMBENT.update(solution=sol,
                                      objective=res["objective"],
                                      obj1=res.get("obj1"))
                    TELEMETRY["construct_objective"] = res["objective"]
                    TELEMETRY["ttci"] = time.monotonic() - t0

        lns_disabled = allow_env and os.environ.get("OGC_DISABLE_LNS") == "1"
        if constructed_ok and not _STOP["flag"] and not lns_disabled:
            saved_env = {}
            if not allow_env:
                saved_env = _suppress_env(("OGC_SEED",))
            try:
                _lns_run(model, prob_info, t0, timelimit_f, base_reserve,
                         [check_cost])
            finally:
                _restore_env(saved_env)

        _STOP["final"] = True
        TELEMETRY["wall"] = time.monotonic() - t0
        return _best_solution()
    except Exception:
        _STOP["final"] = True
        return _best_solution()
    finally:
        if armed:
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev_handler)
            except Exception:
                pass


def _bank_scored_construction(prob_info, total_start, timelimit_f):
    """Return one validated scored-construction incumbent or an empty bank."""
    hard_deadline = min(
        total_start + _HEDGE_BANK_MAX_SECONDS,
        total_start + max(1.0, timelimit_f - 5.0))
    search_deadline = hard_deadline - 0.50
    soft_deadline = search_deadline - _HEDGE_BANK_SOFT_MARGIN_SECONDS
    bank_start = time.monotonic()
    bank = {
        "solution": None,
        "objective": None,
        "obj1": None,
        "elapsed_s": None,
        "construct_s": None,
        "scored_calls": 0,
        "reason": "not-run",
    }
    model = None
    prev_handler = None
    armed = False
    try:
        _STOP.update(flag=False, final=False)
        TELEMETRY.clear()
        _reset_kernel_telemetry("unknown/hedge-bank")
        _cap_thread_pools()
        _mark_validator_available()
        if threading.current_thread() is threading.main_thread():
            try:
                def _on_bank_alarm(signum, frame):
                    raise _DeadlineAlarm()
                prev_handler = signal.signal(signal.SIGALRM, _on_bank_alarm)
                signal.alarm(max(1, math.ceil(
                    hard_deadline - time.monotonic())))
                armed = True
            except Exception:
                prev_handler = None
        model = _Model(prob_info)
        tcon = time.monotonic()
        complete = _construct_scored(model, soft_deadline, search_deadline)
        bank["construct_s"] = time.monotonic() - tcon
        bank["scored_calls"] = TELEMETRY.get("hedge_bank_scored_calls", 0)
        if not complete:
            bank["reason"] = "construct-incomplete"
            return bank
        if time.monotonic() >= hard_deadline:
            bank["reason"] = "bank-deadline"
            return bank
        solution = model.emit()
        result = _validate(prob_info, solution)
        if result is None or not result.get("feasible"):
            bank["reason"] = "validation-failed"
            return bank
        bank.update(
            solution=solution,
            objective=result["objective"],
            obj1=result.get("obj1"),
            reason="valid")
        return bank
    except _DeadlineAlarm:
        bank["reason"] = "bank-alarm"
        return bank
    except Exception as exc:
        bank["reason"] = f"bank-error:{type(exc).__name__}"
        return bank
    finally:
        bank["elapsed_s"] = time.monotonic() - bank_start
        if armed:
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev_handler)
            except Exception:
                pass
        model = None
        gc.collect()
        _STOP.update(flag=False, final=False)


def _candidate1g_armored_algorithm(prob_info, timelimit=60):
    """Submission 7 with a high-pressure scored-construction hedge."""
    total_start = time.monotonic()
    try:
        timelimit_f = float(timelimit)
    except Exception:
        return _submission7_armored_algorithm(prob_info, timelimit=timelimit)
    if timelimit_f < _HEDGE_MIN_TIMELIMIT_SECONDS:
        return _submission7_armored_algorithm(prob_info, timelimit=timelimit)

    route_deadline = min(total_start + 0.50,
                         total_start + max(0.05, 0.01 * timelimit_f))
    ratio = _constructor_pressure_ratio(prob_info, route_deadline)
    _HEDGE_ROUTE_OVERRIDE.update(active=True, ratio=ratio)
    try:
        if ratio is None or ratio < _CONSTRUCTOR_PRESSURE_EDD_THRESHOLD:
            return _submission7_armored_algorithm(
                prob_info, timelimit=timelimit)

        bank = _bank_scored_construction(
            prob_info, total_start, timelimit_f)
        elapsed = time.monotonic() - total_start
        parent_budget = max(1.0, timelimit_f - elapsed)
        parent_solution = _submission7_armored_algorithm(
            prob_info, timelimit=parent_budget)
        parent_objective = _INCUMBENT.get("objective")
        parent_obj1 = _INCUMBENT.get("obj1")

        bank_objective = bank.get("objective")
        if (bank_objective is not None and
                (parent_objective is None or
                 bank_objective < parent_objective)):
            selected = "bank"
            solution = bank["solution"]
            _INCUMBENT.update(solution=solution,
                              objective=bank_objective,
                              obj1=bank.get("obj1"))
        else:
            selected = "submission7"
            solution = parent_solution
            _INCUMBENT.update(solution=parent_solution,
                              objective=parent_objective,
                              obj1=parent_obj1)

        TELEMETRY["hedge_enabled"] = True
        TELEMETRY["hedge_pressure_ratio"] = ratio
        TELEMETRY["hedge_bank_cap_s"] = _HEDGE_BANK_MAX_SECONDS
        TELEMETRY["hedge_bank_elapsed_s"] = bank.get("elapsed_s")
        TELEMETRY["hedge_bank_construct_s"] = bank.get("construct_s")
        TELEMETRY["hedge_bank_scored_calls"] = bank.get("scored_calls")
        TELEMETRY["hedge_bank_reason"] = bank.get("reason")
        TELEMETRY["hedge_bank_objective"] = bank_objective
        TELEMETRY["hedge_parent_budget_s"] = parent_budget
        TELEMETRY["hedge_parent_objective"] = parent_objective
        TELEMETRY["hedge_selected"] = selected
        TELEMETRY["wall"] = time.monotonic() - total_start
        return solution
    finally:
        _HEDGE_ROUTE_OVERRIDE.update(active=False, ratio=None)


def _parallel_race_eligible(prob_info, timelimit_f):
    """Conservative two-model memory and budget gate."""
    try:
        if float(timelimit_f) < _PARALLEL_RACE_MIN_TIMELIMIT_SECONDS:
            return False
        estimate = _armor_l2_estimate(prob_info)
        peak = int(estimate["estimated_peak_bytes"])
        return 2 * peak <= _PARALLEL_RACE_COMBINED_ESTIMATE_LIMIT
    except Exception:
        return False


def _scored_full_armored_algorithm(prob_info, timelimit=60):
    """Full-budget EDD arm with scored fallback in construction and repair."""
    namespace = globals()
    original_construct = namespace["_construct"]
    original_place = namespace["_place_block"]
    original_fast = namespace["_fast_finish_block"]
    original_pressure = namespace["_pressure_finish_block"]

    def scored_place(model, bid, full_scan=False, scored_fallback=False):
        return original_place(
            model, bid, full_scan=full_scan, scored_fallback=True)

    def scored_fast(model, bid, scored_fallback=False):
        return original_fast(model, bid, scored_fallback=True)

    def scored_pressure(model, bid, scored_fallback=False):
        return original_pressure(model, bid, scored_fallback=True)

    def scored_construct(model, deadline):
        TELEMETRY["constructor_order"] = "edd-scored-full"
        return _construct_scored(
            model,
            deadline - _HEDGE_BANK_SOFT_MARGIN_SECONDS,
            deadline,
        )

    namespace["_construct"] = scored_construct
    namespace["_place_block"] = scored_place
    namespace["_fast_finish_block"] = scored_fast
    namespace["_pressure_finish_block"] = scored_pressure
    try:
        return _submission7_armored_algorithm(prob_info, timelimit=timelimit)
    finally:
        namespace["_construct"] = original_construct
        namespace["_place_block"] = original_place
        namespace["_fast_finish_block"] = original_fast
        namespace["_pressure_finish_block"] = original_pressure


def _json_safe(value):
    return json.loads(json.dumps(value, default=str))


def _race_child(arm, prob_info, child_budget, path):
    payload = {"arm": arm, "solution": None, "telemetry": {}, "error": None}
    try:
        if arm == "candidate1g":
            solution = _candidate1g_armored_algorithm(
                prob_info, timelimit=child_budget)
        else:
            solution = _scored_full_armored_algorithm(
                prob_info, timelimit=child_budget)
        payload["solution"] = solution
        payload["telemetry"] = _json_safe(dict(TELEMETRY))
    except BaseException as exc:
        payload["error"] = type(exc).__name__
    try:
        tmp = f"{path}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), default=str)
        os.replace(tmp, path)
    finally:
        os._exit(0)


def _select_race_result(prob_info, results):
    """Officially validate every arm and prefer Candidate 1g on exact ties."""
    best = None
    for item in results:
        solution = item.get("solution")
        if solution is None:
            continue
        try:
            report = _validate(prob_info, solution)
        except Exception:
            report = None
        if report is None or not report.get("feasible"):
            continue
        candidate = dict(item)
        candidate["objective"] = report["objective"]
        candidate["obj1"] = report.get("obj1")
        key = (candidate["objective"],
               0 if candidate.get("arm") == "candidate1g" else 1)
        if best is None or key < best[0]:
            best = (key, candidate)
    return None if best is None else best[1]


def _run_parallel_race(prob_info, timelimit_f, pressure_ratio):
    total_start = time.monotonic()
    fallback = _l1_sparse(prob_info)
    results = [{"arm": "fallback", "solution": fallback, "telemetry": {}}]
    elapsed = time.monotonic() - total_start
    child_budget = max(
        1.0,
        timelimit_f - elapsed - _PARALLEL_RACE_PARENT_RESERVE_SECONDS,
    )
    hard_deadline = total_start + timelimit_f - 0.20
    root = tempfile.mkdtemp(prefix=".ogc-par-race-")
    pids = {}
    paths = {}
    try:
        for arm in ("candidate1g", "scored-full"):
            path = os.path.join(root, f"{arm}.json")
            pid = os.fork()
            if pid == 0:
                _race_child(arm, prob_info, child_budget, path)
            pids[pid] = arm
            paths[arm] = path

        alive = set(pids)
        while alive and time.monotonic() < hard_deadline:
            for pid in tuple(alive):
                try:
                    done, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    done = pid
                if done:
                    alive.discard(pid)
            if alive:
                time.sleep(0.02)

        for pid in tuple(alive):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in tuple(alive):
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass

        for arm, path in paths.items():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    results.append(json.load(handle))
            except Exception:
                results.append({
                    "arm": arm,
                    "solution": None,
                    "telemetry": {},
                    "error": "missing-result",
                })

        selected = _select_race_result(prob_info, results)
        if selected is None:
            selected = {"arm": "fallback", "solution": fallback,
                        "objective": None, "telemetry": {}}

        TELEMETRY.clear()
        TELEMETRY.update({
            "parallel_race_enabled": True,
            "parallel_race_pressure_ratio": pressure_ratio,
            "parallel_race_child_budget_s": child_budget,
            "parallel_race_selected": selected.get("arm"),
            "parallel_race_objective": selected.get("objective"),
            "parallel_race_results": [
                {
                    "arm": item.get("arm"),
                    "error": item.get("error"),
                    "has_solution": item.get("solution") is not None,
                }
                for item in results
            ],
            "wall": time.monotonic() - total_start,
        })
        chosen_tel = selected.get("telemetry") or {}
        TELEMETRY["parallel_race_selected_telemetry"] = chosen_tel
        # Restore required kernel keys after clear() so algorithm() finalize
        # cannot KeyError (WV #2). Prefer selected child telemetry.
        for key, default in _KERNEL_TELEMETRY_DEFAULTS.items():
            if key not in TELEMETRY or TELEMETRY.get(key) is None:
                TELEMETRY[key] = chosen_tel.get(key, default)
        return selected["solution"]
    finally:
        try:
            for name in os.listdir(root):
                os.unlink(os.path.join(root, name))
            os.rmdir(root)
        except Exception:
            pass


def _armored_algorithm(prob_info, timelimit=60):
    total_start = time.monotonic()
    try:
        timelimit_f = float(timelimit)
    except Exception:
        return _candidate1g_armored_algorithm(prob_info, timelimit=timelimit)
    if not hasattr(os, "fork") or not sys.platform.startswith("linux"):
        return _candidate1g_armored_algorithm(prob_info, timelimit=timelimit)
    route_deadline = min(
        total_start + 0.50,
        total_start + max(0.05, 0.01 * timelimit_f),
    )
    ratio = _constructor_pressure_ratio(prob_info, route_deadline)
    if (ratio is None or ratio < _PARALLEL_RACE_PRESSURE_THRESHOLD or
            not _parallel_race_eligible(prob_info, timelimit_f)):
        return _candidate1g_armored_algorithm(prob_info, timelimit=timelimit)
    return _run_parallel_race(prob_info, timelimit_f, ratio)



# ---------------------------------------------------------------------------
# cand_1q3: banded CP-SAT arm (WB H<=113) + sol R8 ARCH E budget split
# Arm absolute hard-kill 6.0s wall (no budget+1.25 slack). At tl=60:
# control guaranteed >=52s; 2.0s parent reserve for setup/check/selection.
# Self-contained; ortools optional with clean fallback.
# ---------------------------------------------------------------------------

_CPSAT_AREA_SCALE = 100
_CPSAT_BAND_HORIZON_MAX = 113
# sol R8 ARCH E frozen split (tl=60): arm<=6.0 absolute | control>=52 | setup=2.0
_CPSAT_LIMIT_S = 2.0           # CP-SAT soft cap (OPTIMAL typically <<1s)
_CPSAT_DECODE_LIMIT_S = 10.0   # decode may use residual arm budget up to this
_CPSAT_ARM_WALL_S = 6.0        # absolute hard-kill wall for whole arm
_CPSAT_CTRL_MIN_S = 52.0       # control guaranteed floor at tl=60
_CPSAT_SETUP_RESERVE_S = 2.0   # parent setup / check / selection reserve
# Back-compat alias (harness/telemetry): residual after arm wall at tl=60
_CPSAT_CTRL_RESERVE_S = _CPSAT_ARM_WALL_S
_CPSAT_SEED = int(os.environ.get("OGC_SEED", "20260615"))

try:
    from ortools.sat.python import cp_model as _cp_model
    _ORTOOLS_OK = True
except Exception:
    _cp_model = None
    _ORTOOLS_OK = False


def _cpsat_poly_area(verts):
    if not verts or len(verts) < 3:
        return 0.0
    a = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def _cpsat_block_area(blk):
    best = None
    for shape in blk["shape"]:
        layers = [L for L in shape["layers"] if L]
        a = 1.0 if not layers else sum(_cpsat_poly_area(L) for L in layers)
        if best is None or a < best:
            best = a
    return float(best if best is not None else 1.0)


def _cpsat_preferred_bay(blk):
    prefs = blk["bay_preferences"]
    return max(range(len(prefs)), key=lambda j: (prefs[j], -j))


def _cpsat_n_layers(blocks):
    k = 1
    for blk in blocks:
        for shape in blk["shape"]:
            k = max(k, len([L for L in shape["layers"] if L]) or 1)
    return k


def _cpsat_instance_structure(prob):
    blocks = prob["blocks"]
    bays = prob["bays"]
    n = len(blocks)
    m = len(bays)
    K = _cpsat_n_layers(blocks)
    bay_areas = [float(b["width"] * b["height"]) for b in bays]
    caps = [int(math.floor(bay_areas[j] * K * _CPSAT_AREA_SCALE)) for j in range(m)]
    R = [_rel_day(blocks[i]) for i in range(n)]
    P = [_dur(blocks[i]) for i in range(n)]
    due = [float(blocks[i]["due_date"]) for i in range(n)]
    areas = [max(1, int(math.ceil(_cpsat_block_area(blocks[i]) * _CPSAT_AREA_SCALE)))
             for i in range(n)]
    bay_of = [_cpsat_preferred_bay(blocks[i]) for i in range(n)]
    max_r = max(R) if R else 0
    max_p = max(P) if P else 1
    max_due = int(max(due)) if due else 0
    by_bay = {j: [] for j in range(m)}
    for i in range(n):
        by_bay[bay_of[i]].append(i)
    extra = 0
    for j in range(m):
        load = sum(areas[i] * P[i] for i in by_bay[j])
        if caps[j] > 0:
            extra = max(extra, int(math.ceil(load / caps[j])) + 5)
    horizon = max(max_due + max_p, max_r + max_p, extra) + 30
    mean_slack = (sum(due[i] - R[i] - P[i] for i in range(n)) / n) if n else 0.0
    total_work = sum(areas[i] * P[i] for i in range(n))
    total_cap = sum(caps) if caps else 1
    rho = total_work / (total_cap * max(1, horizon))
    return {
        "n_blocks": n, "rho": rho, "mean_slack": mean_slack, "horizon": horizon,
        "K": K, "m": m, "R": R, "P": P, "due": due, "areas": areas,
        "bay_of": bay_of, "caps": caps, "by_bay": by_bay,
        "max_r": max_r, "max_p": max_p, "max_due": max_due,
    }


def _cpsat_in_band(feats):
    return int(feats["horizon"]) <= _CPSAT_BAND_HORIZON_MAX


def _cpsat_solve_entry_days(prob, feats, time_limit_s):
    if not _ORTOOLS_OK:
        return {"status": "NO_ORTOOLS", "ok": False, "wall_s": 0.0, "entries": None}
    blocks = prob["blocks"]
    bays = prob["bays"]
    w1 = float(prob.get("weights", {}).get("w1", 1.0))
    n = feats["n_blocks"]
    m = feats["m"]
    R, P, due = feats["R"], feats["P"], feats["due"]
    areas, bay_of, caps = feats["areas"], feats["bay_of"], feats["caps"]
    by_bay = feats["by_bay"]
    horizon = feats["horizon"]
    max_p = feats["max_p"]

    model = _cp_model.CpModel()
    e = [model.NewIntVar(R[i], horizon, f"e_{i}") for i in range(n)]
    t = []
    for i in range(n):
        ti = model.NewIntVar(0, horizon + max_p + 5, f"t_{i}")
        due_i = int(math.floor(due[i] + 1e-9))
        model.Add(ti >= e[i] + P[i] - due_i)
        t.append(ti)

    for j in range(m):
        intervals = []
        demands = []
        for i in by_bay[j]:
            end = model.NewIntVar(R[i] + P[i], horizon + max_p + 5, f"x_{i}")
            model.Add(end == e[i] + P[i])
            iv = model.NewIntervalVar(e[i], P[i], end, f"iv_{i}")
            intervals.append(iv)
            demands.append(areas[i])
        if intervals:
            model.AddCumulative(intervals, demands, caps[j])

    model.Minimize(sum(t))
    solver = _cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = int(_CPSAT_SEED)
    t0 = time.perf_counter()
    status = solver.Solve(model)
    wall = time.perf_counter() - t0
    status_name = solver.StatusName(status)
    ok = status in (_cp_model.OPTIMAL, _cp_model.FEASIBLE)
    entries = [int(solver.Value(e[i])) for i in range(n)] if ok else None
    return {
        "status": status_name, "ok": ok, "wall_s": wall, "entries": entries,
        "sum_tardiness": (sum(int(solver.Value(t[i])) for i in range(n)) if ok else None),
        "obj_w1_proxy": ((w1 * sum(int(solver.Value(t[i])) for i in range(n))) if ok else None),
    }


def _cpsat_place_fast(model, bid, e_target, e_max, deadline):
    """Day-shift placement with feasible_map reuse (WB machinery)."""
    _lazy_imports()
    np = _np
    blk = model.blocks[bid]
    prefs = blk["bay_preferences"]
    bay_order = sorted(range(model.m), key=lambda j: (-prefs[j], j))
    nor = len(blk["shape"])
    R = _rel_day(blk)
    lo = max(e_target, R)
    if lo > e_max:
        return None
    fmap_cache = {}
    NONE = object()

    def fmap(oi, b, e):
        key = (oi, b, e)
        hit = fmap_cache.get(key, NONE)
        if hit is not NONE:
            return hit
        r = model.feasible_map(bid, oi, b, e)
        fmap_cache[key] = r
        return r

    def try_day(e):
        if time.monotonic() > deadline:
            return "timeout"
        for b in bay_order:
            for oi in range(nor):
                r = fmap(oi, b, e)
                if r is None:
                    continue
                feas, x_lo, y_lo = r
                ys, xs = np.nonzero(feas)
                if len(ys) == 0:
                    continue
                if hasattr(model, "contact_map") and len(ys) > 1:
                    xd = e + _dur(blk)
                    try:
                        om = model.om(bid, oi)
                        conv = model.contact_map(bid, oi, b, e, xd)
                        axs = xs + x_lo
                        ays = ys + y_lo
                        contact = model.contact_at(conv, om, axs, ays)
                        i = int(np.argmax(contact))
                        x, y = int(axs[i]), int(ays[i])
                    except Exception:
                        x = int(xs[0] + x_lo)
                        y = int(ys[0] + y_lo)
                else:
                    x = int(xs[0] + x_lo)
                    y = int(ys[0] + y_lo)
                if model.commit(bid, oi, b, x, y, e):
                    return e
        return None

    dense_end = min(e_max, lo + 8)
    for e in range(lo, dense_end + 1):
        got = try_day(e)
        if got == "timeout":
            return None
        if got is not None:
            return got

    e = dense_end + 1
    stride = 2
    prev = dense_end
    while e <= e_max:
        if time.monotonic() > deadline:
            return None
        got = try_day(e)
        if got == "timeout":
            return None
        if got is not None:
            return got
        if stride >= 4:
            mid = (prev + e) // 2
            if mid > prev and mid < e:
                got = try_day(mid)
                if got == "timeout":
                    return None
                if got is not None:
                    return got
        prev = e
        e += stride
        if stride < 16:
            stride *= 2

    for e in range(lo, e_max + 1):
        if time.monotonic() > deadline:
            return None
        got = try_day(e)
        if got == "timeout":
            return None
        if got is not None:
            return got
    return None


def _cpsat_decode_geometry(prob, entries, deadline):
    _reset_kernel_telemetry("cand1q-cpsat-decode")
    model = _Model(prob)
    # Arm decode uses pure-Python feasible_map (WB venue). Kernel path can
    # diverge on contact/placement order and degraded arm objectives on linux.
    model._ckernel_disabled = True
    n = len(prob["blocks"])
    order = sorted(
        range(n),
        key=lambda i: (
            entries[i],
            float(prob["blocks"][i]["due_date"]),
            -float(prob["blocks"][i]["workload"]),
            i,
        ),
    )
    max_entry = max(entries) if entries else 0
    e_cap = max_entry + min(n + 20, 80)
    realized = [None] * n
    shifts = 0
    failed = []
    t0 = time.perf_counter()
    for bid in order:
        if time.monotonic() > deadline:
            failed.append(bid)
            break
        e0 = entries[bid]
        e_got = _cpsat_place_fast(model, bid, e0, e_cap, deadline)
        if e_got is None:
            if time.monotonic() > deadline:
                failed.append(bid)
                continue
            ok = _fast_finish_block(model, bid)
            if not ok or bid not in model.committed:
                failed.append(bid)
                continue
            e_got = model.committed[bid][4]
        realized[bid] = e_got
        if e_got > e0:
            shifts += e_got - e0
    for bid in order:
        if bid in model.committed:
            continue
        if time.monotonic() > deadline:
            break
        ok = _fast_finish_block(model, bid)
        if ok and bid in model.committed:
            realized[bid] = model.committed[bid][4]
            if bid in failed:
                failed.remove(bid)
    wall = time.perf_counter() - t0
    sol = model.emit() if len(model.committed) == n else None
    return {
        "solution": sol,
        "realized_entries": realized,
        "shifts": shifts,
        "n_committed": len(model.committed),
        "failed": failed,
        "decode_wall_s": wall,
    }


def _cpsat_official_objective(prob, solution):
    if not solution or not isinstance(solution, dict):
        return None
    if "operations" not in solution:
        return None
    try:
        rep = _validate(prob, solution)
    except Exception:
        return None
    if not rep or not rep.get("feasible"):
        return None
    obj = rep.get("objective")
    return float(obj) if obj is not None else None


def _cpsat_run_arm_inline(prob, budget_s, t_start):
    """CP-SAT entry-day + decode under budget_s hard wall (in-process).

    Budget split: CP-SAT soft cap, decode residual; outer absolute kill is parent.
    OGC_FORCE_ARM_TIMEOUT=1 holds until outer hard-kill (cert forced-timeout pair).
    """
    feats = _cpsat_instance_structure(prob)
    if os.environ.get("OGC_FORCE_ARM_TIMEOUT", "").strip() in ("1", "true", "TRUE", "yes"):
        # Stay alive past absolute kill wall so parent kill is exercised.
        try:
            time.sleep(max(float(budget_s) + 120.0, 180.0))
        except Exception:
            pass
        return {
            "ok": False, "reason": "forced-timeout", "feats": feats, "killed": True,
            "solution": None, "objective": None,
        }
    remaining = budget_s - (time.monotonic() - t_start)
    if remaining < 1.0:
        return {"ok": False, "reason": "no-budget", "feats": feats, "killed": False}
    # Soft-cap CP-SAT; leave most of the arm wall for decode residual.
    cpsat_lim = min(_CPSAT_LIMIT_S, max(0.5, remaining * 0.35))
    cpsat = _cpsat_solve_entry_days(prob, feats, cpsat_lim)
    out = {
        "ok": False, "feats": feats, "cpsat": cpsat, "decode": None,
        "solution": None, "objective": None, "reason": None, "killed": False,
    }
    if not cpsat["ok"] or not cpsat["entries"]:
        out["reason"] = f"cpsat-{cpsat['status']}"
        return out
    remaining = budget_s - (time.monotonic() - t_start)
    if remaining < 0.5:
        out["reason"] = "decode-no-budget"
        return out
    # Decode uses residual arm budget (up to DECODE_LIMIT); arm hard-kill is outer.
    dec_cap = min(remaining, _CPSAT_DECODE_LIMIT_S)
    dec_deadline = min(t_start + budget_s - 0.10, time.monotonic() + dec_cap)
    dec = _cpsat_decode_geometry(prob, cpsat["entries"], dec_deadline)
    out["decode"] = {
        "shifts": dec["shifts"], "n_committed": dec["n_committed"],
        "failed": dec["failed"], "decode_wall_s": dec["decode_wall_s"],
    }
    if dec["solution"] is None or dec["n_committed"] != feats["n_blocks"]:
        out["reason"] = "decode-incomplete"
        return out
    obj = _cpsat_official_objective(prob, dec["solution"])
    if obj is None:
        out["reason"] = "checker-reject"
        return out
    out["ok"] = True
    out["solution"] = dec["solution"]
    out["objective"] = obj
    out["reason"] = "arm-ok"
    return out


def _cpsat_arm_worker(payload_path, result_path, budget_s):
    """Subprocess entry: hard-killable arm worker (no repo-relative imports)."""
    import json as _json
    with open(payload_path, "r", encoding="utf-8") as fh:
        prob = _json.load(fh)
    t0 = time.monotonic()
    try:
        pack = _cpsat_run_arm_inline(prob, float(budget_s), t0)
    except BaseException as exc:
        pack = {"ok": False, "reason": f"worker-{type(exc).__name__}",
                "solution": None, "objective": None, "killed": False}
    # Strip non-serializable feats for pipe
    safe = {
        "ok": pack.get("ok"),
        "reason": pack.get("reason"),
        "objective": pack.get("objective"),
        "solution": pack.get("solution"),
        "killed": pack.get("killed", False),
        "cpsat_status": (pack.get("cpsat") or {}).get("status") if isinstance(pack.get("cpsat"), dict) else None,
        "cpsat_wall": (pack.get("cpsat") or {}).get("wall_s") if isinstance(pack.get("cpsat"), dict) else None,
        "decode": pack.get("decode"),
        "horizon": (pack.get("feats") or {}).get("horizon") if isinstance(pack.get("feats"), dict) else None,
    }
    tmp = result_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(safe, fh, separators=(",", ":"), default=str)
    os.replace(tmp, result_path)


def _run_cpsat_arm_hard_kill(prob, budget_s):
    """Hard-kill arm at budget_s ABSOLUTE wall via subprocess (sol R8 ARCH E).

    kill means killed at budget_s wall, period — no +1.25s wait slack.
    """
    if not _ORTOOLS_OK:
        return {"ok": False, "reason": "no-ortools", "killed": False,
                "solution": None, "objective": None}
    root = tempfile.mkdtemp(prefix=".ogc-cpsat-arm-")
    payload = os.path.join(root, "prob.json")
    result = os.path.join(root, "result.json")
    with open(payload, "w", encoding="utf-8") as fh:
        json.dump(prob, fh, separators=(",", ":"))
    # Spawn worker importing this module by path (self-contained file only).
    here = os.path.abspath(__file__)
    worker_src = (
        "import importlib.util, sys, os\n"
        f"spec = importlib.util.spec_from_file_location('cand_1q3_arm', {here!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['cand_1q3_arm'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        f"mod._cpsat_arm_worker({payload!r}, {result!r}, {float(budget_s)!r})\n"
    )
    worker_py = os.path.join(root, "worker.py")
    with open(worker_py, "w", encoding="utf-8") as fh:
        fh.write(worker_src)
    env = os.environ.copy()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = "1"
    env["OGC_SEED"] = str(_CPSAT_SEED)
    # Propagate forced-timeout cert flag into the arm worker.
    if os.environ.get("OGC_FORCE_ARM_TIMEOUT"):
        env["OGC_FORCE_ARM_TIMEOUT"] = os.environ["OGC_FORCE_ARM_TIMEOUT"]
    # Ensure utils import path for checker inside worker if needed
    try:
        import utils as _u  # noqa: F401
        utils_dir = os.path.dirname(os.path.abspath(_u.__file__))
        env["PYTHONPATH"] = utils_dir + os.pathsep + env.get("PYTHONPATH", "")
    except Exception:
        pass
    import subprocess
    # sol R8 ARCH E: absolute kill at budget_s — no budget+1.25 wait defect.
    kill_after = max(0.05, float(budget_s))
    t0 = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, worker_py],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        proc.wait(timeout=kill_after)
        killed = False
    except subprocess.TimeoutExpired:
        killed = True
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    wall = time.monotonic() - t0
    pack = {"ok": False, "reason": "arm-no-result", "killed": killed,
            "solution": None, "objective": None, "arm_wall_s": wall}
    try:
        if os.path.isfile(result):
            with open(result, "r", encoding="utf-8") as fh:
                pack = json.load(fh)
            pack["killed"] = killed or pack.get("killed", False)
            pack["arm_wall_s"] = wall
    except Exception:
        pack = {"ok": False, "reason": "arm-read-fail", "killed": killed,
                "solution": None, "objective": None, "arm_wall_s": wall}
    finally:
        try:
            for name in os.listdir(root):
                try:
                    os.unlink(os.path.join(root, name))
                except Exception:
                    pass
            os.rmdir(root)
        except Exception:
            pass
    return pack


def algorithm(prob_info, timelimit=60):
    """Always armored (no instance-identity routing). Banded CP-SAT arm with
    absolute 6.0s hard-kill + control >=52s @ tl=60 + 2.0s setup reserve;
    return better checker-valid sol. Any arm failure falls through to control.
    """
    t0 = time.monotonic()
    try:
        timelimit_f = float(timelimit)
    except Exception:
        timelimit_f = 60.0

    if "OGC_SEED" not in os.environ:
        os.environ["OGC_SEED"] = str(_CPSAT_SEED)

    diag = {
        "cand_1q3": True,
        "band": False,
        "horizon": None,
        "path": None,
        "arm": None,
        "control_objective": None,
        "control_budget_s": None,
        "selected_objective": None,
        "ortools_ok": _ORTOOLS_OK,
        "arm_wall_cap_s": _CPSAT_ARM_WALL_S,
        "ctrl_min_s": _CPSAT_CTRL_MIN_S,
        "setup_reserve_s": _CPSAT_SETUP_RESERVE_S,
        "ctrl_reserve_s": _CPSAT_CTRL_RESERVE_S,
    }

    arm_pack = None
    band = False
    # ALL arm orchestration (structure/setup/spawn/solve/decode) is wrapped:
    # any exception -> clean control-only path.
    try:
        feats = _cpsat_instance_structure(prob_info)
        band = _cpsat_in_band(feats)
        diag["band"] = band
        diag["horizon"] = feats["horizon"]
        diag["n_blocks"] = feats["n_blocks"]
        diag["rho"] = feats["rho"]

        if band and _ORTOOLS_OK:
            # sol R8 ARCH E: arm absolute <=6.0s; leave control >=52s @ tl=60
            # and 2.0s parent setup/check/selection reserve.
            # arm_budget = min(6, tl - 52 - 2) = 6 at tl=60.
            arm_budget = min(
                _CPSAT_ARM_WALL_S,
                max(
                    0.0,
                    timelimit_f - _CPSAT_CTRL_MIN_S - _CPSAT_SETUP_RESERVE_S,
                ),
            )
            if arm_budget >= 1.0:
                arm_pack = _run_cpsat_arm_hard_kill(prob_info, arm_budget)
                diag["arm"] = {
                    "ok": arm_pack.get("ok"),
                    "reason": arm_pack.get("reason"),
                    "objective": arm_pack.get("objective"),
                    "cpsat_status": arm_pack.get("cpsat_status"),
                    "cpsat_wall": arm_pack.get("cpsat_wall"),
                    "killed": arm_pack.get("killed"),
                    "arm_wall_s": arm_pack.get("arm_wall_s"),
                    "decode": arm_pack.get("decode"),
                    "budget_s": arm_budget,
                }
            else:
                diag["arm"] = {
                    "ok": False, "reason": "arm-budget-too-small",
                    "killed": False, "budget_s": arm_budget,
                }
        elif band and not _ORTOOLS_OK:
            diag["arm"] = {"ok": False, "reason": "no-ortools", "killed": False}
    except Exception as exc:
        arm_pack = None
        # Keep band flag if structure succeeded before later arm failure;
        # if structure itself failed, band stays False.
        diag["arm"] = {
            "ok": False,
            "reason": f"arm-exception:{type(exc).__name__}",
            "killed": False,
        }
        diag["arm_exception"] = type(exc).__name__

    # Dominance guarantee: ALWAYS run armored control on remaining budget.
    # Absolute arm hard-kill ensures arm cannot starve control of residual.
    # 2.0s setup/check/selection reserve kept off the control budget.
    outer_cap = max(1.0, timelimit_f - _CPSAT_SETUP_RESERVE_S)
    elapsed = time.monotonic() - t0
    if band:
        # Residual after arm absolute kill; at tl=60 residual >=52 under nominal.
        ctrl_budget = max(1.0, outer_cap - elapsed)
    else:
        # Out of band: full (capped) budget on armored control path.
        ctrl_budget = outer_cap
    ctrl_sol = _armored_algorithm(prob_info, timelimit=ctrl_budget)
    ctrl_obj = _cpsat_official_objective(prob_info, ctrl_sol)
    diag["control_objective"] = ctrl_obj
    diag["control_budget_s"] = ctrl_budget

    candidates = []
    if (arm_pack and arm_pack.get("ok") and arm_pack.get("solution") is not None
            and arm_pack.get("objective") is not None):
        candidates.append(
            ("cpsat-arm", arm_pack["solution"], float(arm_pack["objective"]))
        )
    if ctrl_sol is not None and ctrl_obj is not None:
        candidates.append(("control", ctrl_sol, float(ctrl_obj)))

    if not candidates:
        solution = ctrl_sol if ctrl_sol is not None else {"operations": {}}
        diag["path"] = "control-fallback-empty"
    elif not band:
        # Out of band: pure armored control (no arm competed).
        solution = ctrl_sol if ctrl_sol is not None else {"operations": {}}
        diag["path"] = "control-exact"
        if ctrl_obj is not None:
            diag["selected_objective"] = ctrl_obj
    else:
        # Selection = better checker-valid of (arm, control); ties -> control.
        best = min(candidates, key=lambda x: (x[2], 0 if x[0] == "control" else 1))
        solution = best[1]
        diag["path"] = best[0]
        diag["selected_objective"] = best[2]

    # Stash control kernel telemetry before finalize strips non-kernel keys.
    # _armored_algorithm / race path already populated TELEMETRY.
    _finalize_kernel_telemetry()
    # Re-surface cand_1q3 diagnostics for harness (submission-harmless extras).
    TELEMETRY.update({k: v for k, v in diag.items() if v is not None})
    TELEMETRY["wall"] = time.monotonic() - t0
    return solution
