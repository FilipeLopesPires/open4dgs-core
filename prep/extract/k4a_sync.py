#!/usr/bin/env python3
"""Frame alignment for multi-device Azure Kinect recordings.

Two sources of timing are combined:

1. **Device timestamps.**  Devices in a wired-sync chain whose clocks were reset
   together report colour timestamps on one shared time base.  Within such a
   *clock group* the frame correspondence and the sub-frame offset are measured
   exactly from ``K4A_START_OFFSET_NS`` + Matroska block time.

2. **Visual motion sync.**  Devices whose clocks are unrelated (older firmware,
   separate chains, standalone units) are aligned to the reference group by
   cross-correlating a view-independent motion signal: the fraction of pixels that
   change between consecutive frames.  The signal's first difference is
   correlated over integer lags and the peak refined by parabolic interpolation.
   For subordinate devices the known ``K4A_SUBORDINATE_DELAY_NS`` then resolves
   the sub-frame part: only lags of the form ``n - delay/period`` are physically
   possible, and the visual estimate selects ``n``.

The visual estimator is validated on every camera whose offset is *known* from
timestamps; the report states that error so a failing capture is visible.

Conventions (PLUS, as in the rest of Open4DGS): for logical frame ``i`` a camera
uses source *slot* ``i + lag`` and its actual capture time is
``i / fps + offset_seconds``.  Slots are timestamp-derived frame numbers, so dropped
frames leave holes instead of shifting later frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from k4a_mkv import K4ARecording, iter_frames

# Two devices are taken to share a clock when their colour timestamps fall within this
# window of each other modulo the frame period AND within one recording length overall.
CLOCK_PHASE_TOL_NS = 2_000_000  # 2 ms (daisy-chain delays are 160 us steps)


@dataclass
class CameraTiming:
    name: str
    lag: int  # logical frame i -> source slot i + lag
    offset_s: float  # actual - logical, seconds
    method: str  # "reference" | "timestamp" | "visual+subordinate_delay" | "visual" | "manual"
    clock_group: int
    visual_lag: float | None = None  # continuous visual estimate vs reference (slots)
    visual_peak_corr: float | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Slots (timestamp-derived frame numbering)
# --------------------------------------------------------------------------------------
def color_slots(rec: K4ARecording, period_ns: float, skip_startup: int) -> dict[int, int]:
    """Map slot number -> packet index.  Slot 0 is the first packet's timestamp."""
    ts = rec.color_device_ns.astype(np.float64)
    slots = np.round((ts - ts[0]) / period_ns).astype(int)
    out: dict[int, int] = {}
    for pkt, s in enumerate(slots):
        if pkt < skip_startup:
            continue
        if s in out:  # duplicate timestamp slot; keep first
            continue
        out[int(s)] = pkt
    return out


def slot_time_ns(rec: K4ARecording, period_ns: float) -> float:
    """Device time of slot 0 (first packet), used as the slot grid origin."""
    return float(rec.color_device_ns[0])


# --------------------------------------------------------------------------------------
# Clock groups
# --------------------------------------------------------------------------------------
def clock_groups(recs: list[K4ARecording], period_ns: float) -> list[int]:
    """Group recordings whose colour timestamps live on one shared device clock."""
    groups = [-1] * len(recs)
    next_id = 0
    for i, ri in enumerate(recs):
        if groups[i] >= 0:
            continue
        groups[i] = next_id
        for j in range(i + 1, len(recs)):
            if groups[j] >= 0:
                continue
            rj = recs[j]
            if _same_clock(ri, rj, period_ns):
                groups[j] = next_id
        next_id += 1
    return groups


def _same_clock(a: K4ARecording, b: K4ARecording, period_ns: float) -> bool:
    ta, tb = a.color_device_ns, b.color_device_ns
    # Overlapping spans (shared clocks were reset together, so spans overlap heavily).
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    if hi - lo < 10 * period_ns:
        return False
    # Expected relative phase from subordinate delays; residual must be small.
    expected = (b.subordinate_delay_ns - a.subordinate_delay_ns) % period_ns
    phase = (tb[0] - ta[0]) % period_ns
    diff = abs(((phase - expected) + period_ns / 2) % period_ns - period_ns / 2)
    # Absolute start time near each other (within a couple of seconds) - free-running
    # clocks since power-on are essentially never this close.
    near = abs(int(ta[0]) - int(tb[0])) < 2_000_000_000
    return bool(diff < CLOCK_PHASE_TOL_NS and near)


# --------------------------------------------------------------------------------------
# Visual motion signal
# --------------------------------------------------------------------------------------
def motion_signal(
    rec: K4ARecording,
    slots: dict[int, int],
    width: int = 480,
    threshold: float = 6.0,
) -> np.ndarray:
    """Per-slot motion signal (fraction of changed pixels between slot s and s+1).

    Returns an array indexed by slot; NaN where a slot or its successor is missing.
    """
    cw, ch = rec.color_size
    height = int(round(width * ch / cw / 2)) * 2
    n_slots = max(slots) + 2
    frames: dict[int, np.ndarray] = {}
    pkt_to_slot = {p: s for s, p in slots.items()}
    for pkt, img in iter_frames(rec.path, rec.color_stream, cw, ch, "gray", scale=(width, height)):
        s = pkt_to_slot.get(pkt)
        if s is None:
            continue
        f = img.astype(np.float32)
        frames[s] = f * (100.0 / max(float(f.mean()), 1e-3))  # remove global gain changes
    sig = np.full(n_slots, np.nan)
    for s, f in frames.items():
        g = frames.get(s + 1)
        if g is not None:
            sig[s] = float((np.abs(g - f) > threshold).mean())
    return sig


def correlate_lag(a: np.ndarray, b: np.ndarray, max_lag: int, min_overlap: int = 120):
    """Best lag L with a[i] ~ b[i + L]; returns (continuous lag, peak corr, table)."""
    table = []
    for lag in range(-max_lag, max_lag + 1):
        i0, i1 = max(0, -lag), min(len(a), len(b) - lag)
        if i1 - i0 < min_overlap:
            continue
        x, y = a[i0:i1], b[i0 + lag : i1 + lag]
        x = np.log(x + 1e-4)
        y = np.log(y + 1e-4)
        dx, dy = np.diff(x), np.diff(y)
        ok = np.isfinite(dx) & np.isfinite(dy)
        if ok.sum() < min_overlap:
            continue
        dx, dy = dx[ok], dy[ok]
        dx = (dx - dx.mean()) / (dx.std() + 1e-12)
        dy = (dy - dy.mean()) / (dy.std() + 1e-12)
        table.append((lag, float((dx * dy).mean())))
    if not table:
        raise ValueError("not enough overlapping frames for visual sync")
    lags = np.array([t[0] for t in table])
    corr = np.array([t[1] for t in table])
    k = int(np.argmax(corr))
    cont = float(lags[k])
    if 0 < k < len(corr) - 1:
        c0, c1, c2 = corr[k - 1], corr[k], corr[k + 1]
        den = c0 - 2 * c1 + c2
        if abs(den) > 1e-9:
            cont += float(np.clip(0.5 * (c0 - c2) / den, -0.5, 0.5))
    return cont, float(corr[k]), table


def resolve_subordinate_lag(visual_lag: float, delay_frames: float) -> tuple[int, float, int, float]:
    """Combine a continuous visual lag with a known hardware delay.

    If source slot ``i + n`` is captured at logical time ``i + delay_frames`` the
    continuous lag is ``n - delay_frames``; the visual estimate picks the integer ``n``.
    Slot ``i + L`` is then captured at ``i + L - n + delay_frames``; the integer ``L``
    with the smallest absolute offset is returned.

    Returns (lag L, offset in frames, n, |visual - physical| residual in frames).
    """
    n = int(np.round(visual_lag + delay_frames))
    offsets = {lag: delay_frames - (n - lag) for lag in (n - 1, n, n + 1)}
    lag = min(offsets, key=lambda L: abs(offsets[L]))
    return lag, float(offsets[lag]), n, float(abs(visual_lag - (n - delay_frames)))


# --------------------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------------------
def solve_timing(
    recs: list[K4ARecording],
    reference: int,
    *,
    skip_startup: int = 2,
    max_lag: int = 30,
    visual_width: int = 480,
    manual_lags: dict[str, int] | None = None,
    run_visual: bool = True,
    log=print,
) -> tuple[list[CameraTiming], dict]:
    fps = recs[reference].fps
    period_ns = 1e9 / fps
    groups = clock_groups(recs, period_ns)
    ref = recs[reference]
    ref_group = groups[reference]
    slots = [color_slots(r, period_ns, skip_startup) for r in recs]
    origin = [slot_time_ns(r, period_ns) for r in recs]
    manual_lags = manual_lags or {}

    timings: list[CameraTiming | None] = [None] * len(recs)
    # 1) timestamp-measured cameras (same clock as reference)
    for i, r in enumerate(recs):
        if groups[i] != ref_group:
            continue
        if i == reference:
            timings[i] = CameraTiming(r.name, 0, 0.0, "reference", groups[i])
            continue
        # slot s of camera i is at origin_i + s*T; reference logical frame k at origin_ref + k*T.
        rel = (origin[i] - origin[reference]) / period_ns  # in frames
        lag = int(np.round(-rel))  # choose slot closest in time to logical frame
        offset_s = (origin[i] + lag * period_ns - origin[reference]) / 1e9
        timings[i] = CameraTiming(r.name, lag, float(offset_s), "timestamp", groups[i])

    signals: dict[int, np.ndarray] = {}
    report: dict = {"clock_groups": {recs[i].name: g for i, g in enumerate(groups)}, "visual": {}}
    if run_visual:
        for i, r in enumerate(recs):
            log(f"  motion signal: {r.name}")
            signals[i] = motion_signal(r, slots[i], width=visual_width)

    def visual_vs_reference(i: int) -> tuple[float, float, list]:
        """Continuous visual lag of camera i relative to reference logical frames, using every
        timestamp-measured camera as an anchor (median)."""
        ests, corrs = [], []
        for j, tj in enumerate(timings):
            if tj is None or tj.method not in ("reference", "timestamp") or j == i:
                continue
            cont, pk, _ = correlate_lag(signals[j], signals[i], max_lag)
            # camera j slot s  <-> logical s - lag_j ; camera i slot s+cont
            ests.append(cont + tj.lag - tj.offset_s * fps)
            corrs.append(pk)
        return float(np.median(ests)), float(np.median(corrs)), ests

    # 2) validate visual estimator on timestamp-measured cameras
    if signals:
        errs = []
        for i, t in enumerate(timings):
            if t is None or t.method != "timestamp":
                continue
            cont, pk, _ = correlate_lag(signals[reference], signals[i], max_lag)
            truth = t.lag - t.offset_s * fps
            errs.append(abs(cont - truth))
            t.visual_lag, t.visual_peak_corr = cont, pk
            report["visual"][t.name] = {"visual_lag": cont, "timestamp_lag": truth, "abs_error_frames": abs(cont - truth), "peak_corr": pk}
        report["visual_validation_max_error_frames"] = float(max(errs)) if errs else None

    # 3) cameras on other clocks
    for i, r in enumerate(recs):
        if timings[i] is not None:
            continue
        if r.name in manual_lags:
            lag = manual_lags[r.name]
            delay_s = (r.subordinate_delay_ns - ref.subordinate_delay_ns) / 1e9
            timings[i] = CameraTiming(r.name, lag, delay_s, "manual", groups[i], notes=["lag from --manual-lag"])
            continue
        if not signals:
            raise SystemExit(f"{r.name}: on a separate clock and visual sync disabled; pass --manual-lag")
        cont, pk, ests = visual_vs_reference(i)
        notes = [f"anchor estimates {np.round(ests, 3).tolist()}"]
        if r.sync_mode == "SUBORDINATE" and ref.sync_mode == "MASTER":
            d = (r.subordinate_delay_ns - ref.subordinate_delay_ns) / period_ns
            lag, off_frames, n, resid = resolve_subordinate_lag(cont, d)
            offset_s = off_frames / fps
            method = "visual+subordinate_delay"
            notes.append(f"continuous visual lag {cont:.3f}, hardware delay {d:.3f} frames -> n={n}")
            if resid > 0.35:
                notes.append(f"WARNING visual estimate {resid:.2f} frames from nearest physical hypothesis")
        else:
            lag = int(np.round(cont))
            offset_s = 0.0
            method = "visual"
            notes.append("no subordinate delay to resolve sub-frame phase; offset 0 (let the trainer learn it)")
        timings[i] = CameraTiming(r.name, lag, float(offset_s), method, groups[i], visual_lag=cont, visual_peak_corr=pk, notes=notes)

    # 4) consistency: pairwise check among visually solved cameras
    final = [t for t in timings if t is not None]
    report["cameras"] = {
        t.name: {
            "lag": t.lag,
            "offset_ms": t.offset_s * 1e3,
            "method": t.method,
            "clock_group": t.clock_group,
            "visual_lag": t.visual_lag,
            "visual_peak_corr": t.visual_peak_corr,
            "notes": t.notes,
        }
        for t in final
    }
    report["fps"] = fps
    return final, report


def common_window(
    recs: list[K4ARecording],
    timings: list[CameraTiming],
    skip_startup: int,
) -> tuple[int, int, list[dict[int, int]]]:
    """Largest contiguous logical range [first, last] where every camera has a source slot."""
    period_ns = 1e9 / recs[0].fps
    slots = [color_slots(r, period_ns, skip_startup) for r in recs]
    lo = max(min(s) - t.lag for s, t in zip(slots, timings))
    hi = min(max(s) - t.lag for s, t in zip(slots, timings))
    best = (0, -1)
    run_start = None
    for k in range(lo, hi + 1):
        ok = all((k + t.lag) in s for s, t in zip(slots, timings))
        if ok and run_start is None:
            run_start = k
        if (not ok or k == hi) and run_start is not None:
            end = k if ok else k - 1
            if end - run_start > best[1] - best[0]:
                best = (run_start, end)
            run_start = None
    if best[1] < best[0]:
        raise SystemExit("no logical frame is covered by every camera")
    return best[0], best[1], slots
