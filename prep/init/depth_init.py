#!/usr/bin/env python3
"""Dense, metric TriGS point initialisation from multi-view depth.

Writes the same NPZ schema as ``prep-roma-trigs-memory.py`` (consumed by
``trigs-train-colmap --npz-path``), but the points come from the depth of every camera
instead of RoMa matches.  Depth is read through the source-agnostic depth sources of
``prep/mask/depth_masks.py`` (``--depth-source auto|registered|k4a``); nothing here is
specific to one sensor.  With ``--depth-offsets-json`` (output of
``prep/extract/k4a_depth_offsets.py``) each camera's additive depth correction is added to
the colour-camera z of its valid depth at load; the offsets are recorded in the NPZ
(``depth_offsets_m``, ``depth_offsets_cameras``).

Per keyframe t (every ``--keyframe-step`` frames of ``--frame-start..--frame-end``):

1. Every camera's depth is z-buffered into that camera's colour image at
   ``--work-scale`` (default: half resolution).  Single-pixel gaps are filled where the
   neighbours agree.
2. Flying pixels are rejected: a pixel is dropped when any pixel within
   ``--edge-radius`` differs in depth by more than ``--edge-abs + --edge-rel * z``, or
   when it borders a hole (``--hole-erode``).
3. Pixels inside the camera's own mask (>= ``--mask-min-fg-alpha``) are unprojected to
   world space.  A point is kept when at least ``--min-votes`` cameras see it (it projects
   inside the image and is not occluded: that camera's depth is not more than
   ``--occlusion-margin`` in front of it) and at least ``--mask-min-fg-fraction`` of them
   vote foreground.  Detached 3D clusters are then dropped (``--cluster-voxel``, the
   rule of ``depth_masks.main_cluster``).
4. Colour = per-channel median over the cameras that see the point (depth within
   ``--colour-margin``); the source camera always counts.
5. Velocity: dense DIS optical flow per camera between t and t+d (and t-d with
   ``--velocity-mode centred``; d = ``--velocity-delta``, default half the keyframe
   step), forward-backward checked, lifted through the depth at both ends, divided by d.
   World units per video frame.
6. With a sync file, each camera's points are moved from its capture time to the
   nominal frame time (``X - v * offset_frames``), the same convention as prep-roma's
   motion-compensated rows; times stay on the keyframe grid.
7. A voxel grid sized to give at most ``--target-points`` occupied voxels fuses all
   candidates: centroid position, mean colour, per-component median velocity of the
   samples that have one.  Voxels without a velocity inherit the median of measured
   neighbours within ``--velocity-inherit-radius``; then a k-NN median smooths the field.

Every row is a dynamic keyframe row (``is_static`` False, ``dynamic_allowed`` and
``stage_keep`` True), like the masked prep-roma output.  Gaussian scales are not part of
the schema: the trainer derives them from k-NN spacing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# The per-camera worker threads call numpy matmuls concurrently (here and in
# depth_masks).  A multi-threaded OpenBLAS called from several Python threads hung or
# crashed intermittently on Windows, so BLAS is pinned to one thread.  This only works
# before numpy is first imported; otherwise main() falls back to --workers 1.
_BLAS_PINNED = "numpy" not in sys.modules
if _BLAS_PINNED:
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mask"))
import depth_masks as dm  # noqa: E402  (load_cameras, make_source, to_world, CameraPose)

TRACK_FWD = np.uint8(1)  # same bits as prep/init/rigid_motion.py
TRACK_BWD = np.uint8(2)


# ======================================================================================
# Geometry helpers
# ======================================================================================
@dataclass
class WorkCam:
    """A calibrated camera plus its working-resolution grid."""

    cam: dm.CameraPose
    scale: float
    w: int = 0
    h: int = 0
    rays: np.ndarray = field(default=None, repr=False)  # [h, w, 2] normalised x/z, y/z

    def __post_init__(self):
        self.w = max(1, int(round(self.cam.width * self.scale)))
        self.h = max(1, int(round(self.cam.height * self.scale)))
        uu, vv = np.meshgrid(np.arange(self.w, dtype=np.float64), np.arange(self.h, dtype=np.float64))
        self.rays = self.normalised(uu.ravel(), vv.ravel()).reshape(self.h, self.w, 2)

    # work pixel <-> full-resolution pixel (pixel centres aligned)
    def to_full(self, u, v):
        sx, sy = self.cam.width / self.w, self.cam.height / self.h
        return (np.asarray(u) + 0.5) * sx - 0.5, (np.asarray(v) + 0.5) * sy - 0.5

    def to_work(self, uf, vf):
        sx, sy = self.cam.width / self.w, self.cam.height / self.h
        return (np.asarray(uf) + 0.5) / sx - 0.5, (np.asarray(vf) + 0.5) / sy - 0.5

    def normalised(self, u, v) -> np.ndarray:
        """Undistorted normalised coordinates of (float) work pixels."""
        uf, vf = self.to_full(u, v)
        pix = np.stack([uf, vf], 1).astype(np.float64)
        if not np.any(self.cam.dist != 0):
            K = self.cam.K
            return np.stack([(pix[:, 0] - K[0, 2]) / K[0, 0], (pix[:, 1] - K[1, 2]) / K[1, 1]], 1).astype(np.float32)
        return cv2.undistortPoints(pix[:, None], self.cam.K, self.cam.dist)[:, 0].astype(np.float32)


def project_full(cam: dm.CameraPose, Xw: np.ndarray):
    """Full-resolution pixel coordinates (float) and depth; aligned with the input rows."""
    Xc = Xw.astype(np.float64) @ cam.R.T + cam.t
    z = Xc[:, 2]
    front = z > 0.05
    uv = np.full((len(Xw), 2), -1e9)
    if front.any():
        if np.any(cam.dist != 0):
            p, _ = cv2.projectPoints(Xc[front], np.zeros(3), np.zeros(3), cam.K, cam.dist)
            uv[front] = p[:, 0]
        else:
            zf = z[front]
            uv[front, 0] = cam.K[0, 0] * Xc[front, 0] / zf + cam.K[0, 2]
            uv[front, 1] = cam.K[1, 1] * Xc[front, 1] / zf + cam.K[1, 2]
    return uv, z, front


def zbuffer_depth(wc: WorkCam, pts_cam: np.ndarray) -> np.ndarray:
    """Nearest z-depth per work pixel from colour-camera-frame points; 0 = no depth."""
    D = np.zeros((wc.h, wc.w), np.float32)
    if len(pts_cam) == 0:
        return D
    cam = wc.cam
    z = pts_cam[:, 2].astype(np.float64)
    ok = z > 0.05
    P, z = pts_cam[ok].astype(np.float64), z[ok]
    if np.any(cam.dist != 0):
        p, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), cam.K, cam.dist)
        uf, vf = p[:, 0, 0], p[:, 0, 1]
    else:
        uf = cam.K[0, 0] * P[:, 0] / z + cam.K[0, 2]
        vf = cam.K[1, 1] * P[:, 1] / z + cam.K[1, 2]
    u, v = wc.to_work(uf, vf)
    u = np.round(u).astype(np.int64)
    v = np.round(v).astype(np.int64)
    inb = (u >= 0) & (u < wc.w) & (v >= 0) & (v < wc.h)
    idx, z = v[inb] * wc.w + u[inb], z[inb]
    order = np.argsort(z, kind="stable")
    uniq, first = np.unique(idx[order], return_index=True)
    D.ravel()[uniq] = z[order][first]
    return D


def fill_small_holes(D: np.ndarray, min_neighbours: int, abs_thr: float, rel_thr: float) -> np.ndarray:
    """Fill empty pixels whose 8 neighbours are mostly valid and agree in depth."""
    valid = D > 0
    k = np.ones((3, 3), np.float32)
    s = cv2.filter2D(np.where(valid, D, 0).astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    n = cv2.filter2D(valid.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    mean = s / np.maximum(n, 1)
    dmax = cv2.dilate(np.where(valid, D, 0).astype(np.float32), np.ones((3, 3), np.uint8))
    dmin = cv2.erode(np.where(valid, D, 1e6).astype(np.float32), np.ones((3, 3), np.uint8))
    fill = (~valid) & (n >= min_neighbours) & ((dmax - dmin) <= abs_thr + rel_thr * mean)
    out = D.copy()
    out[fill] = mean[fill]
    return out


def edge_ok(D: np.ndarray, radius: int, abs_thr: float, rel_thr: float, hole_erode: int) -> np.ndarray:
    """Valid depth away from depth discontinuities (flying pixels) and hole borders."""
    valid = D > 0
    ok = valid.copy()
    if radius > 0:
        ker = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        dmax = cv2.dilate(np.where(valid, D, 0).astype(np.float32), ker)
        dmin = cv2.erode(np.where(valid, D, 1e6).astype(np.float32), ker)
        thr = abs_thr + rel_thr * D
        ok &= (dmax - D <= thr) & (D - dmin <= thr)
    if hole_erode > 0:
        ker = np.ones((2 * hole_erode + 1, 2 * hole_erode + 1), np.uint8)
        ok &= cv2.erode(valid.astype(np.uint8), ker, borderType=cv2.BORDER_REPLICATE) > 0
    return ok


def sample_bilinear(img: np.ndarray, uf: np.ndarray, vf: np.ndarray) -> np.ndarray:
    """Bilinear sample of an [H, W, C] image at float pixel coordinates (clamped)."""
    H, W = img.shape[:2]
    x = np.clip(uf, 0, W - 1.001)
    y = np.clip(vf, 0, H - 1.001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    ax = (x - x0)[:, None]
    ay = (y - y0)[:, None]
    I = img.astype(np.float32)
    return ((1 - ax) * (1 - ay) * I[y0, x0] + ax * (1 - ay) * I[y0, x0 + 1]
            + (1 - ax) * ay * I[y0 + 1, x0] + ax * ay * I[y0 + 1, x0 + 1])


# ======================================================================================
# Fusion helpers
# ======================================================================================
def voxel_groups(pos: np.ndarray, voxel: float):
    q = np.floor((pos - pos.min(0)) / voxel).astype(np.int64)
    dims = q.max(0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    return inv.reshape(-1), int(inv.max()) + 1


def choose_voxel(pos: np.ndarray, target: int, iters: int = 30) -> float:
    """Smallest voxel size whose occupied-voxel count is <= target (bisection, log scale)."""
    if len(pos) <= target:
        return 1e-4
    ext = float(np.linalg.norm(pos.max(0) - pos.min(0))) + 1e-6
    lo, hi = 1e-4, ext
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        n = voxel_groups(pos, mid)[1]
        if n > target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.002:
            break
    return hi


def group_mean(groups: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    cnt = np.bincount(groups, minlength=n).astype(np.float64)
    out = np.stack([np.bincount(groups, weights=values[:, d], minlength=n) for d in range(values.shape[1])], 1)
    return (out / np.maximum(cnt, 1)[:, None]).astype(np.float32)


def group_median(groups: np.ndarray, values: np.ndarray, valid: np.ndarray, n: int):
    """Per-group, per-component median of the valid rows. Returns (median [n, D], count [n])."""
    g = groups[valid]
    vals = values[valid]
    cnt = np.bincount(g, minlength=n)
    out = np.zeros((n, values.shape[1]), np.float32)
    has = cnt > 0
    if not has.any():
        return out, cnt
    for d in range(values.shape[1]):
        order = np.lexsort((vals[:, d], g))
        gs, vs = g[order], vals[order, d]
        start = np.searchsorted(gs, np.arange(n))
        lo = start + (cnt - 1) // 2
        hi = start + cnt // 2
        out[has, d] = 0.5 * (vs[lo[has]] + vs[hi[has]])
    return out, cnt


def inherit_velocity(pos: np.ndarray, vel: np.ndarray, has: np.ndarray, radius: float, k: int = 8):
    """Rows without velocity take the median of up to k measured neighbours within radius."""
    vel, has = vel.copy(), has.copy()
    miss = np.flatnonzero(~has)
    if len(miss) == 0 or not has.any() or radius <= 0:
        return vel, has, 0
    src = np.flatnonzero(has)
    tree = cKDTree(pos[src])
    kk = min(k, len(src))
    d, j = tree.query(pos[miss], k=kk, distance_upper_bound=radius)
    d, j = d.reshape(len(miss), kk), j.reshape(len(miss), kk)
    found = np.isfinite(d)
    nb = np.where(found[..., None], vel[src[np.minimum(j, len(src) - 1)]], np.nan)
    ok = found.any(1)
    vel[miss[ok]] = np.nanmedian(nb[ok], axis=1)
    has[miss[ok]] = True
    return vel, has, int(ok.sum())


def smooth_velocity(pos: np.ndarray, vel: np.ndarray, has: np.ndarray, k: int) -> np.ndarray:
    """k-NN (self included) per-component median over rows with a velocity."""
    if k <= 1 or has.sum() <= 1:
        return vel
    src = np.flatnonzero(has)
    kk = min(k, len(src))
    _, j = cKDTree(pos[src]).query(pos[src], k=kk)
    out = vel.copy()
    out[src] = np.median(vel[src][j.reshape(len(src), kk)], axis=1)
    return out


def scale_duration_by_speed(vel, has, base, min_scale, max_scale):
    """Same rule as prep-roma-trigs-memory.py: base * clip(1 / (1 + speed / median), min, max)."""
    scales = np.full(len(vel), max_scale, np.float32)
    if has.any():
        sp = np.linalg.norm(vel, axis=1)
        med = float(np.median(sp[has]))
        if med > 0:
            s = np.clip(1.0 / (1.0 + sp / med), min_scale, max_scale)
            scales[has] = s[has]
    return (scales * base).reshape(-1, 1).astype(np.float32)


# ======================================================================================
# Inputs
# ======================================================================================
def load_sync_offsets(path: Path | None, cams, fps: float) -> np.ndarray:
    """Per-camera capture offsets in video frames from a {camera-key: seconds} JSON."""
    off = np.zeros(len(cams))
    if path is None:
        return off
    raw = {str(k): float(v) for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items()}
    missing = []
    for i, c in enumerate(cams):
        keys = [c.folder, c.folder.lstrip("0") or "0"]
        hit = next((k for k in keys if k in raw), None)
        if hit is None:
            missing.append(c.folder)
        else:
            off[i] = raw[hit] * fps
    if missing:
        print(f"[sync] WARNING: no offset for cameras {missing}; using 0")
    return off


def load_depth_offsets(path: Path, cams) -> np.ndarray:
    """Per-camera additive depth corrections (metres) from a ``k4a_depth_offsets.py`` JSON
    (``offsets_m`` keyed by camera folder; a plain {folder: metres} object also works)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    table = raw.get("offsets_m", raw) if isinstance(raw, dict) else None
    if not isinstance(table, dict):
        raise SystemExit(f"{path}: expected an 'offsets_m' object keyed by camera folder")
    missing = [c.folder for c in cams if c.folder not in table]
    if missing:
        raise SystemExit(f"{path}: no depth offset for cameras {missing}")
    return np.array([float(table[c.folder]) for c in cams], np.float64)


class OffsetDepthSource:
    """Wraps a depth source and adds a per-camera offset to the colour-camera z of every
    valid depth point (the point moves along its colour-camera ray; invalid depth was
    never a point, so it stays invalid)."""

    def __init__(self, source, offsets: np.ndarray):
        self.source = source
        self.offsets = np.asarray(offsets, np.float64)
        self.name = f"{source.name}+offsets"

    def points_cam(self, ci: int, frame: int) -> np.ndarray:
        P = self.source.points_cam(ci, frame)
        o = float(self.offsets[ci])
        if o == 0.0 or len(P) == 0:
            return P
        z = P[:, 2].astype(np.float64)
        keep = (z > 0) & (z + o > 0)
        s = (z[keep] + o) / z[keep]
        return (P[keep].astype(np.float64) * s[:, None]).astype(np.float32)


class Inputs:
    def __init__(self, args, cams, source):
        self.a = args
        self.cams = cams
        self.source = source
        self.wcs = [WorkCam(c, args.work_scale) for c in cams]
        self.images_root = Path(args.data_dir) / args.images_root
        self.masks_root = Path(args.data_dir) / args.masks_root if args.masks_root else None
        self.dis = None

    def image(self, ci, frame):
        p = self.images_root / self.cams[ci].folder / self.a.image_pattern.format(frame=frame)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"missing image {p}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def gray_work(self, rgb, wc):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(g, (wc.w, wc.h), interpolation=cv2.INTER_AREA)

    def mask(self, ci, frame):
        if self.masks_root is None:
            c = self.cams[ci]
            return np.ones((c.height, c.width), np.float32)
        p = self.masks_root / self.cams[ci].folder / self.a.mask_pattern.format(frame=frame)
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"missing mask {p}")
        c = self.cams[ci]
        if m.shape != (c.height, c.width):
            m = cv2.resize(m, (c.width, c.height), interpolation=cv2.INTER_LINEAR)
        return m.astype(np.float32) / 255.0

    def depth(self, ci, frame):
        """(filled z-buffer depth, edge-filtered validity) at work resolution."""
        a = self.a
        wc = self.wcs[ci]
        D = zbuffer_depth(wc, self.source.points_cam(ci, frame))
        if a.fill_holes:
            D = fill_small_holes(D, 5, a.edge_abs, a.edge_rel)
        return D, edge_ok(D, a.edge_radius, a.edge_abs, a.edge_rel, a.hole_erode)


def make_dis(preset: str):
    presets = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST, "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
               "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
    return cv2.DISOpticalFlow_create(presets[preset])


def lift_flow(wc: WorkCam, rows, cols, d0, F, B, D1, ok1, fb_max, max_dz):
    """World positions at the other frame of the pixels (rows, cols) moved by flow F.
    Returns (X1 [N,3] world, valid [N])."""
    n = len(rows)
    X1 = np.zeros((n, 3), np.float32)
    u = cols + F[rows, cols, 0]
    v = rows + F[rows, cols, 1]
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    valid = (ui >= 0) & (ui < wc.w) & (vi >= 0) & (vi < wc.h)
    uic, vic = np.clip(ui, 0, wc.w - 1), np.clip(vi, 0, wc.h - 1)
    fb = F[rows, cols] + B[vic, uic]
    valid &= np.hypot(fb[:, 0], fb[:, 1]) <= fb_max
    d1 = D1[vic, uic]
    valid &= ok1[vic, uic] & (np.abs(d1 - d0) <= max_dz)
    if valid.any():
        xy = wc.normalised(u[valid], v[valid])
        z = d1[valid][:, None]
        X1[valid] = dm.to_world(wc.cam, np.concatenate([xy * z, z], 1))
    return X1, valid


# ======================================================================================
# One keyframe
# ======================================================================================
def camera_candidates(inp: Inputs, ci: int, t: int, nbr: dict, cache: dict):
    """Foreground candidates of camera ci at frame t with their per-camera motion."""
    a = inp.a
    wc = inp.wcs[ci]
    D0, ok0 = cache[(ci, t)]["depth"]
    mw = cv2.resize(cache[(ci, t)]["mask"], (wc.w, wc.h), interpolation=cv2.INTER_AREA)
    sel = ok0 & (mw >= a.mask_min_fg_alpha)
    rows, cols = np.nonzero(sel)
    d0 = D0[rows, cols]
    xy = wc.rays[rows, cols]
    X0 = dm.to_world(wc.cam, np.concatenate([xy * d0[:, None], d0[:, None]], 1))
    out = {"X": X0, "rows": rows, "cols": cols, "vf": None, "vb": None}
    dis = make_dis(a.flow_preset)
    g0 = cache[(ci, t)]["gray"]
    for key, f1 in nbr.items():
        g1 = cache[(ci, f1)]["gray"]
        F = dis.calc(g0, g1, None)
        B = dis.calc(g1, g0, None)
        D1, ok1 = cache[(ci, f1)]["depth"]
        X1, val = lift_flow(wc, rows, cols, d0, F, B, D1, ok1, a.fb_max_px, a.max_dz * abs(f1 - t))
        out[key] = (X1, val)
    return out


def process_keyframe(inp: Inputs, t: int, frame_lo: int, frame_hi: int, sync_off: np.ndarray, pool):
    a = inp.a
    nC = len(inp.cams)
    timing = {}
    t0 = time.perf_counter()
    nbr = {}
    dt = int(a.velocity_delta) if int(a.velocity_delta) > 0 else max(1, (int(a.keyframe_step) + 1) // 2)
    if t + dt <= frame_hi:
        nbr["vf"] = t + dt
    if (a.velocity_mode == "centred" or "vf" not in nbr) and t - dt >= frame_lo:
        nbr["vb"] = t - dt
    frames = [t] + list(nbr.values())

    cache = {}

    def load(ci_f):
        ci, f = ci_f
        rgb = inp.image(ci, f)
        entry = {"gray": inp.gray_work(rgb, inp.wcs[ci]), "depth": inp.depth(ci, f)}
        if f == t:
            entry["rgb"] = rgb
            entry["mask"] = inp.mask(ci, f)
        return (ci, f), entry

    for k, v in pool.map(load, [(ci, f) for ci in range(nC) for f in frames]):
        cache[k] = v
    timing["load_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    per_cam = list(pool.map(lambda ci: camera_candidates(inp, ci, t, nbr, cache), range(nC)))
    timing["flow_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    X = np.concatenate([c["X"] for c in per_cam]).astype(np.float32)
    src = np.concatenate([np.full(len(c["X"]), ci, np.int32) for ci, c in enumerate(per_cam)])
    n_cand = len(X)

    def motion(key):
        V = np.zeros((n_cand, 3), np.float32)
        ok = np.zeros(n_cand, bool)
        Xo = np.zeros((n_cand, 3), np.float32)
        if key in nbr:
            Xo = np.concatenate([c[key][0] for c in per_cam])
            ok = np.concatenate([c[key][1] for c in per_cam])
            V = (Xo - X) if key == "vf" else (X - Xo)
            V = V / float(abs(nbr[key] - t))
        return V, ok, Xo

    Vf, okf, X1 = motion("vf")
    Vb, okb, Xm1 = motion("vb")

    # multi-view foreground vote and colour, using every camera's depth at t
    fg = np.zeros(n_cand, np.int32)
    votes = np.zeros(n_cand, np.int32)
    col_sum = []
    for k in range(nC):
        wc = inp.wcs[k]
        uv, z, front = project_full(wc.cam, X)
        inb = front & (uv[:, 0] >= 0) & (uv[:, 0] <= wc.cam.width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= wc.cam.height - 1)
        uw, vw = wc.to_work(uv[:, 0], uv[:, 1])
        uwi = np.clip(np.round(uw).astype(np.int64), 0, wc.w - 1)
        vwi = np.clip(np.round(vw).astype(np.int64), 0, wc.h - 1)
        Dk = cache[(k, t)]["depth"][0][vwi, uwi]
        occluded = (Dk > 0) & (Dk < z - a.occlusion_margin)
        vote = inb & ~occluded
        mk = cache[(k, t)]["mask"]
        ui = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, wc.cam.width - 1)
        vi = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, wc.cam.height - 1)
        fg += (vote & (mk[vi, ui] >= a.mask_min_fg_alpha)).astype(np.int32)
        votes += vote.astype(np.int32)
        sees = (inb & (Dk > 0) & (np.abs(Dk - z) <= a.colour_margin)) | (src == k)
        c = np.full((n_cand, 3), np.nan, np.float32)
        if sees.any():
            c[sees] = sample_bilinear(cache[(k, t)]["rgb"], uv[sees, 0], uv[sees, 1]) / 255.0
        col_sum.append(c)
    keep = votes >= a.min_votes
    if inp.masks_root is not None:
        keep &= fg >= np.ceil(a.mask_min_fg_fraction * np.maximum(votes, 1) - 1e-9)
    n_after_vote = int(keep.sum())
    if a.cluster_voxel > 0 and keep.any():
        # same rule as depth_masks: largest 3D cluster plus clusters >= keep-ratio of it
        idx = np.flatnonzero(keep)
        keep[idx[~dm.main_cluster(X[idx], a.cluster_voxel, a.cluster_keep_ratio)]] = False
    n_views = np.sum([~np.isnan(c[:, 0]) for c in col_sum], axis=0)
    C = np.nanmedian(np.stack(col_sum, 1)[keep], axis=1).astype(np.float32)
    X, src, Vf, okf, Vb, okb, X1, Xm1 = (arr[keep] for arr in (X, src, Vf, okf, Vb, okb, X1, Xm1))
    timing["vote_colour_s"] = time.perf_counter() - t2

    # per-candidate velocity (centred where both ends exist) and acceleration
    t3 = time.perf_counter()
    both = okf & okb
    V = np.where(both[:, None], 0.5 * (Vf + Vb), np.where(okf[:, None], Vf, Vb)).astype(np.float32)
    okv = okf | okb
    A = ((X1 - 2.0 * X + Xm1) / float(dt * dt)).astype(np.float32)  # per video frame^2
    if a.max_speed > 0:
        okv &= np.linalg.norm(V, axis=1) <= a.max_speed
    # capture time -> nominal frame time
    if np.any(sync_off != 0):
        o = sync_off[src].astype(np.float32)[:, None]
        X = np.where(okv[:, None], X - V * o, X).astype(np.float32)

    if len(X) == 0:
        print(f"  frame {t}: WARNING no foreground depth points survived; keyframe left empty")
        e3, e0 = np.zeros((0, 3), np.float32), np.zeros(0, bool)
        info = {"frame": int(t), "flow_frames": {k: int(v) for k, v in nbr.items()}, "candidates": int(n_cand),
                "after_vote": n_after_vote, "after_cluster": 0, "colour_views_mean": 0.0, "voxel_m": 0.0,
                "points": 0, "velocity_measured": 0, "velocity_inherited": 0, "velocity_zero": 0,
                "speed_p50": 0.0, "speed_p90": 0.0, "speed_p99": 0.0, "speed_max": 0.0,
                "timing": {"total_s": round(time.perf_counter() - t0, 2)}}
        return dict(positions=e3, colors=e3, velocities=e3, has_velocity=e0, accelerations=e3,
                    has_acceleration=e0, track_flags=np.zeros(0, np.uint8)), info

    # voxel fusion to the target count
    voxel = choose_voxel(X, a.target_points)
    g, n = voxel_groups(X, voxel)
    P = group_mean(g, X, n)
    Cg = np.clip(group_mean(g, C, n), 0, 1)
    Vg, cnt = group_median(g, V, okv, n)
    Ag, cnta = group_median(g, A, both, n)
    has = cnt >= a.min_velocity_samples
    measured = int(has.sum())
    Vg[~has] = 0.0
    flags = ((np.bincount(g[okf], minlength=n) > 0) * TRACK_FWD
             | (np.bincount(g[okb], minlength=n) > 0) * TRACK_BWD).astype(np.uint8)
    Vg, has, inherited = inherit_velocity(P, Vg, has, a.velocity_inherit_radius)
    Vg = smooth_velocity(P, Vg, has, a.velocity_smooth_k)
    Vg[~has] = 0.0
    has_acc = cnta > 0
    Ag[~has_acc] = 0.0
    timing["fuse_s"] = time.perf_counter() - t3
    timing["total_s"] = time.perf_counter() - t0

    sp = np.linalg.norm(Vg[has], axis=1) if has.any() else np.zeros(1)
    info = {
        "frame": int(t), "flow_frames": {k: int(v) for k, v in nbr.items()},
        "candidates": int(n_cand), "after_vote": n_after_vote, "after_cluster": int(keep.sum()),
        "colour_views_mean": float(n_views[keep].mean()) if keep.any() else 0.0,
        "voxel_m": float(voxel), "points": int(n),
        "velocity_measured": measured, "velocity_inherited": inherited, "velocity_zero": int((~has).sum()),
        "speed_p50": float(np.percentile(sp, 50)), "speed_p90": float(np.percentile(sp, 90)),
        "speed_p99": float(np.percentile(sp, 99)), "speed_max": float(sp.max()),
        "timing": {k: round(v, 2) for k, v in timing.items()},
    }
    return dict(positions=P, colors=Cg, velocities=Vg.astype(np.float32), has_velocity=has,
                accelerations=Ag.astype(np.float32), has_acceleration=has_acc, track_flags=flags), info


# ======================================================================================
# NPZ assembly
# ======================================================================================
ROW_KEYS = ("positions", "colors", "velocities", "accelerations", "has_acceleration", "track_flags",
            "times", "durations", "has_velocity", "is_static", "dynamic_protect", "dynamic_allowed",
            "stage_keep", "source_frames")


def keyframe_rows(kf: dict, t: int, frame_start: int, denom: int, step: int, a) -> dict:
    n = len(kf["positions"])
    base = step / float(denom)
    if a.speed_adaptive_duration:
        dur = scale_duration_by_speed(kf["velocities"], kf["has_velocity"], base, a.duration_min_scale, a.duration_max_scale)
    else:
        dur = np.full((n, 1), base, np.float32)
    return dict(
        positions=kf["positions"].astype(np.float32), colors=kf["colors"].astype(np.float32),
        velocities=kf["velocities"].astype(np.float32), accelerations=kf["accelerations"].astype(np.float32),
        has_acceleration=kf["has_acceleration"].astype(bool), track_flags=kf["track_flags"].astype(np.uint8),
        times=np.full((n, 1), (t - frame_start) / float(denom), np.float32), durations=dur,
        has_velocity=kf["has_velocity"].astype(bool), is_static=np.zeros(n, bool),
        dynamic_protect=np.zeros(n, bool), dynamic_allowed=np.ones(n, bool), stage_keep=np.ones(n, bool),
        source_frames=np.full((n, 1), t, np.int32),
    )


def merge_rows(path: Path, keyframes, frame_start: int, frame_end: int, denom: int, fps: float) -> dict:
    """Rows of another prep NPZ (e.g. prep-roma) at our keyframes, re-timed to our range."""
    z = np.load(path, allow_pickle=False)
    if str(z["velocity_units"]) != "world_per_video_frame":
        raise ValueError(f"{path}: velocity_units {z['velocity_units']} (need world_per_video_frame)")
    if "fps" in z.files and abs(float(z["fps"]) - fps) > 1e-6:
        raise ValueError(f"{path}: fps {float(z['fps'])} != --fps {fps}")
    sf = np.asarray(z["source_frames"]).reshape(-1)
    static = np.asarray(z["is_static"]).astype(bool) if "is_static" in z.files else np.zeros(len(sf), bool)
    sel = static | np.isin(sf, np.asarray(keyframes))
    n = int(sel.sum())
    their_denom = max(int(z["source_frame_end"]) - int(z["source_frame_start"]), 1)
    out = {}
    for k in ROW_KEYS:
        if k in z.files:
            out[k] = np.asarray(z[k])[sel]
    defaults = dict(accelerations=np.zeros((n, 3), np.float32), has_acceleration=np.zeros(n, bool),
                    track_flags=np.zeros(n, np.uint8), has_velocity=np.ones(n, bool), is_static=np.zeros(n, bool),
                    dynamic_protect=np.zeros(n, bool), dynamic_allowed=np.ones(n, bool), stage_keep=np.ones(n, bool))
    for k, v in defaults.items():
        out.setdefault(k, v)
    out["times"] = ((sf[sel] - frame_start) / float(denom)).reshape(-1, 1).astype(np.float32)
    out["times"][out["is_static"]] = 0.5
    if "durations" in out:
        out["durations"] = (out["durations"].astype(np.float64) * their_denom / denom).astype(np.float32)
    else:
        out["durations"] = np.full((n, 1), 1.0 / denom, np.float32)
    out["source_frames"] = sf[sel].reshape(-1, 1).astype(np.int32)
    print(f"[merge] {path.name}: {n:,} rows ({int(out['is_static'].sum()):,} static) at our keyframes")
    return out


def write_ply(path: Path, pos: np.ndarray, col: np.ndarray, vel: np.ndarray | None = None) -> None:
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if vel is not None:
        fields += [("vx", "<f4"), ("vy", "<f4"), ("vz", "<f4")]
    arr = np.empty(len(pos), dtype=fields)
    arr["x"], arr["y"], arr["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    c8 = np.clip(np.round(col * 255), 0, 255).astype(np.uint8)
    arr["red"], arr["green"], arr["blue"] = c8[:, 0], c8[:, 1], c8[:, 2]
    if vel is not None:
        arr["vx"], arr["vy"], arr["vz"] = vel[:, 0], vel[:, 1], vel[:, 2]
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(pos)}"]
    head += [f"property {'float' if t == '<f4' else 'uchar'} {n}" for n, t in fields] + ["end_header"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(arr.tobytes())


# ======================================================================================
# CLI
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("inputs")
    g.add_argument("--data-dir", default=".")
    g.add_argument("--reference-json", default="reference_export.json")
    g.add_argument("--images-root", default="images")
    g.add_argument("--image-pattern", default="{frame:06d}.jpg")
    g.add_argument("--masks-root", default="masks", help="'' or 'none' disables masks (not recommended)")
    g.add_argument("--mask-pattern", default="{frame:06d}.png")
    g.add_argument("--depth-source", default="auto", choices=["auto", "registered", "k4a"])
    g.add_argument("--depth-dir", default=None, help="default: depth_registered (registered) / depth (k4a)")
    g.add_argument("--depth-scale", type=float, default=0.001, help="registered PNG metres per unit")
    g.add_argument("--depth-offsets-json", default=None,
                   help="per-camera additive depth corrections (prep/extract/k4a_depth_offsets.py output): "
                        "offsets_m[camera] is added to the colour-camera z of every valid depth point")
    g.add_argument("--sync-json", default="sync.json",
                   help="per-camera capture offsets (s); used when the file exists. 'none' disables")
    g.add_argument("--fps", type=float, default=30.0)
    g = p.add_argument_group("frames / output")
    g.add_argument("--frame-start", type=int, default=None, help="default: first common image frame")
    g.add_argument("--frame-end", type=int, default=None, help="inclusive; default: last common image frame")
    g.add_argument("--keyframe-step", type=int, default=5)
    g.add_argument("--output-path", required=True)
    g.add_argument("--target-points", type=int, default=60000, help="max points per keyframe (voxel fusion)")
    g.add_argument("--merge-npz", default=None, help="append another prep NPZ's rows at the same keyframes")
    g.add_argument("--export-ply-dir", default=None, help="write depth_init_f<frame>.ply per keyframe")
    g.add_argument("--stats-path", default=None)
    g.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    g = p.add_argument_group("depth filtering")
    g.add_argument("--work-scale", type=float, default=0.5, help="depth/flow grid size vs the colour image")
    g.add_argument("--no-fill-holes", dest="fill_holes", action="store_false")
    g.add_argument("--edge-radius", type=int, default=1, help="window radius (work px) of the discontinuity test")
    g.add_argument("--edge-abs", type=float, default=0.03, help="m")
    g.add_argument("--edge-rel", type=float, default=0.01, help="x depth")
    g.add_argument("--hole-erode", type=int, default=1, help="drop pixels this close (work px) to a hole")
    g = p.add_argument_group("foreground / colour")
    g.add_argument("--mask-min-fg-alpha", type=float, default=0.5)
    g.add_argument("--mask-min-fg-fraction", type=float, default=0.75)
    g.add_argument("--min-votes", type=int, default=2,
                   help="cameras (source included) that must see a point unoccluded; 1 keeps unverifiable points")
    g.add_argument("--occlusion-margin", type=float, default=0.05,
                   help="m; a camera whose depth is this much in front of the point does not vote")
    g.add_argument("--cluster-voxel", type=float, default=0.05,
                   help="m; keep the largest 26-connected voxel cluster (+ clusters >= keep-ratio of it); 0 = off")
    g.add_argument("--cluster-keep-ratio", type=float, default=0.15)
    g.add_argument("--colour-margin", type=float, default=0.03, help="m; depth agreement for a colour view")
    g = p.add_argument_group("velocity")
    g.add_argument("--velocity-mode", default="centred", choices=["centred", "forward"],
                   help="centred: mean of t-1->t and t->t+1 where both exist")
    g.add_argument("--velocity-delta", type=int, default=0,
                   help="frames between t and the flow partner(s), velocity divided by it; "
                        "0 = (keyframe-step + 1) // 2 (depth noise falls as 1 / delta)")
    g.add_argument("--flow-preset", default="medium", choices=["ultrafast", "fast", "medium"])
    g.add_argument("--fb-max-px", type=float, default=1.0, help="forward-backward flow check (work px)")
    g.add_argument("--max-dz", type=float, default=0.08, help="max depth change per frame along a flow track (m)")
    g.add_argument("--max-speed", type=float, default=0.1, help="m per frame; faster samples are discarded")
    g.add_argument("--min-velocity-samples", type=int, default=1, help="per voxel")
    g.add_argument("--velocity-inherit-radius", type=float, default=0.05, help="m")
    g.add_argument("--velocity-smooth-k", type=int, default=8, help="k-NN median smoothing (1 = off)")
    g.add_argument("--no-speed-adaptive-duration", dest="speed_adaptive_duration", action="store_false")
    g.add_argument("--duration-min-scale", type=float, default=0.2)
    g.add_argument("--duration-max-scale", type=float, default=1.0)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.workers > 1 and not _BLAS_PINNED:
        print("[depth_init] numpy was imported before depth_init: BLAS threads not pinned, using --workers 1")
        a.workers = 1
    data = Path(a.data_dir)
    if a.masks_root in ("", "none", "None"):
        a.masks_root = None
    cams = dm.load_cameras(data, a.reference_json)
    source = dm.make_source(a.depth_source, data, cams, a.depth_dir, a.depth_scale)
    depth_offsets = None
    if a.depth_offsets_json:
        op = Path(a.depth_offsets_json)
        op = op if op.is_absolute() or op.exists() else data / op
        depth_offsets = load_depth_offsets(op, cams)
        source = OffsetDepthSource(source, depth_offsets)
        print(f"[depth_init] depth offsets from {op} (mm, added to valid depth): "
              + ", ".join(f"{c.folder}={o * 1000:+.1f}" for c, o in zip(cams, depth_offsets)))
    if a.frame_start is None or a.frame_end is None:
        f0, f1 = dm.detect_frame_range(data / a.images_root, cams, Path(a.image_pattern).suffix.lstrip("."))
        a.frame_start = f0 if a.frame_start is None else a.frame_start
        a.frame_end = f1 if a.frame_end is None else a.frame_end
    if a.frame_end < a.frame_start:
        raise SystemExit("--frame-end < --frame-start")
    keyframes = list(range(a.frame_start, a.frame_end + 1, a.keyframe_step))
    denom = max(a.frame_end - a.frame_start, 1)
    sync_path = None if a.sync_json in (None, "", "none") else data / a.sync_json
    if sync_path is not None and not sync_path.exists():
        sync_path = None
    sync_off = load_sync_offsets(sync_path, cams, a.fps)
    print(f"[depth_init] {len(cams)} cameras, depth source '{source.name}', frames {a.frame_start}..{a.frame_end}, "
          f"{len(keyframes)} keyframes (step {a.keyframe_step}), target {a.target_points:,} points/keyframe")
    print(f"[depth_init] sync offsets (frames): "
          + (", ".join(f"{c.folder}={o:+.3f}" for c, o in zip(cams, sync_off)) if sync_path else "none"))

    inp = Inputs(a, cams, source)
    rows, infos = [], []
    t_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        for t in keyframes:
            kf, info = process_keyframe(inp, t, a.frame_start, a.frame_end, sync_off, pool)
            infos.append(info)
            rows.append(keyframe_rows(kf, t, a.frame_start, denom, a.keyframe_step, a))
            print(f"  frame {t:6d}: {info['candidates']:,} candidates -> {info['after_vote']:,} after vote -> "
                  f"{info['after_cluster']:,} after cluster -> "
                  f"{info['points']:,} points (voxel {info['voxel_m'] * 1000:.1f} mm); velocity measured "
                  f"{info['velocity_measured']:,}, inherited {info['velocity_inherited']:,}, zero "
                  f"{info['velocity_zero']:,}; |v| p50/p90/max {info['speed_p50'] * 1000:.1f}/"
                  f"{info['speed_p90'] * 1000:.1f}/{info['speed_max'] * 1000:.1f} mm/frame; {info['timing']['total_s']:.1f} s")
            if a.export_ply_dir:
                write_ply(Path(a.export_ply_dir) / f"depth_init_f{t:06d}.ply", kf["positions"], kf["colors"], kf["velocities"])

    if a.merge_npz:
        rows.append(merge_rows(Path(a.merge_npz), keyframes, a.frame_start, a.frame_end, denom, a.fps))
    arrays = {k: np.concatenate([r[k] for r in rows]) for k in ROW_KEYS}
    n = len(arrays["positions"])
    out = Path(a.output_path)
    if not out.is_absolute():
        out = data / out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    extra = {}
    if depth_offsets is not None:
        extra = dict(depth_offsets_m=depth_offsets.astype(np.float32),
                     depth_offsets_cameras=np.array([c.folder for c in cams]))
    np.savez_compressed(
        tmp, **arrays, **extra,
        frame_start=np.array(0, np.int32), frame_end=np.array(a.frame_end - a.frame_start, np.int32),
        n_keyframes=np.array(len(keyframes), np.int32), keyframe_step=np.array(a.keyframe_step, np.int32),
        source_frame_start=np.array(a.frame_start, np.int32), source_frame_end=np.array(a.frame_end, np.int32),
        keyframe_frames=np.asarray(keyframes, np.int32), fps=np.array(a.fps, np.float32),
        velocity_units=np.array("world_per_video_frame"),
        time_units=np.array("normalized_0_1_over_source_frame_range"),
        mode=np.array("depth_init_dynamic" + ("_merged" if a.merge_npz else "")),
    )
    os.replace(tmp, out)
    elapsed = time.perf_counter() - t_all
    hv = int(arrays["has_velocity"].sum())
    print(f"Saved {n:,} TriGS NPZ points across {len(keyframes)} keyframes to {out}")
    print(f"Velocity units: world_per_video_frame; has velocity {hv:,} / {n:,} ({100.0 * hv / max(n, 1):.1f}%)")
    print(f"Static rows: {int(arrays['is_static'].sum()):,}; total time {elapsed:.1f} s "
          f"({elapsed / max(len(keyframes), 1):.1f} s/keyframe)")
    if a.stats_path:
        sp = Path(a.stats_path)
        sp = sp if sp.is_absolute() else data / sp
        sp.write_text(json.dumps({"args": vars(a), "depth_source": source.name,
                                  "depth_offsets_m": None if depth_offsets is None else
                                  {c.folder: float(o) for c, o in zip(cams, depth_offsets)},
                                  "keyframes": infos,
                                  "rows": n, "seconds": round(elapsed, 1)}, indent=1, default=str) + "\n",
                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
