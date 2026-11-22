#!/usr/bin/env python3
"""Per-camera additive depth corrections from cross-camera depth consistency.

Time-of-flight cameras carry a per-device depth bias (Azure Kinect: several mm to ~2 cm).
On a calibrated rig the bias shows up as a constant disagreement between cameras that see
the same surface.  This script measures that disagreement on the subject and solves for
one additive correction per camera::

    depth_corrected = depth + offsets_m[camera]      (valid pixels only; 0 stays invalid)

Input is the generic registered-depth contract (``depth_registered/<cam>/<frame:06d>.png``,
uint16, ``--depth-scale`` metres per unit, z-depth of the colour camera; ``.npy`` float
metres also read), ``reference_export.json`` and any mask directory (255 = subject).  Nothing
is Kinect-specific except the name.

Method, per sampled frame and ordered camera pair (i, j):

1. Pixels of camera i that are valid, away from depth discontinuities (``--edge-radius``,
   ``--edge-rel``, ``--edge-abs``) and holes (``--hole-erode``), and inside camera i's mask
   eroded by ``--mask-erode`` depth pixels, are lifted to world space (up to
   ``--max-points`` per camera).
2. They are projected into camera j.  Where j's depth is also edge-safe and inside j's
   eroded mask (bilinear over 4 such pixels), the residual r = z_j(point) - d_j(measured) is
   taken.  Gate: |r| < ``--gate-wide``, then |r - median| < ``--gate``.  The pair counts
   when at least ``--min-points`` residuals survive.
3. Per pair, the frame medians are combined by their median over frames (robust to
   motion and sync).  With corrections o, the pair residual becomes
   r + c_ij o_i - o_j, where c_ij = mean of (n.ray_i) / (n.ray_j) over the points (surface
   normal n, rays with unit z in their camera; ~1 for surfaces facing both cameras).
   Least squares over all pairs gives o (optionally Huber
   IRLS, ``--huber-mm``).  Only pairs whose surfaces face both cameras
   (``--max-view-angle``) are measured, which in practice keeps neighbouring cameras.

Gauge: a common shift of all cameras changes the pair residuals only through
(c_ij - 1), i.e. through the calibrated geometry, so it is weakly identified and
confounded with the rig scale.  The fit always includes it (so it cannot leak into the
differences); ``--gauge mean`` (default) then removes it: the corrections have mean zero,
only relative offsets are applied and the average depth level (to which the calibration
scale was fitted) is kept.  ``--gauge free`` keeps it.
As an independent check, the common shift is also estimated from a colour-triangulated
point set (``--gauge-points``: an ``.npy`` [N,3] world point array, default
``k4a_report/calibration_points.npy`` when present, or a COLMAP ``points3D.txt``): per
camera the median of (camera z of the point - measured depth there), minus the mean-gauge
offset.  It is reported, never applied.

Output (``--output``, default ``<data-dir>/depth_offsets.json``)::

    {"contract": "metres to ADD to every valid depth value of the camera (z along the colour
                  optical axis)",
     "offsets_m": {"0001": ..., ...}, "gauge": "mean" | "free",
     "frames": [...], "pair_residual_rms_mm": {"before": ..., "after": ...},
     "per_pair": {...}, "per_frame_offsets_mm": {...}, "per_frame_spread_mm": {...}, ...}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mask"))
import depth_masks as dm  # noqa: E402  (load_cameras, to_world, CameraPose)

CONTRACT = "metres to ADD to every valid depth value of the camera (z along the colour optical axis)"


# ======================================================================================
# Inputs
# ======================================================================================
def load_depth(root: Path, folder: str, frame: int, scale: float) -> np.ndarray:
    """Registered z-depth in metres (0 = invalid) from <root>/<folder>/<frame:06d>.png|.npy."""
    base = root / folder / f"{frame:06d}"
    if base.with_suffix(".npy").exists():
        d = np.load(base.with_suffix(".npy")).astype(np.float32)
    else:
        img = cv2.imread(str(base.with_suffix(".png")), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"missing depth {base}.png/.npy")
        if img.ndim == 3:
            img = img[..., 0]
        d = img.astype(np.float32) * scale
    d[~np.isfinite(d)] = 0
    return d


def load_mask(root: Path, folder: str, frame: int, shape: tuple[int, int]) -> np.ndarray:
    """Mask in [0, 1] resampled (area) to the depth grid."""
    p = root / folder / f"{frame:06d}.png"
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"missing mask {p}")
    return cv2.resize(m.astype(np.float32) / 255.0, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def frames_in(root: Path, cams) -> set[int]:
    common = None
    for c in cams:
        d = root / c.folder
        got = {int(m.group(1)) for p in d.iterdir() if (m := re.fullmatch(r"(\d{6})\.(png|npy)", p.name))} \
            if d.is_dir() else set()
        common = got if common is None else common & got
    return common or set()


def pick_frames(available: set[int], n: int) -> list[int]:
    """n frames spread evenly over the available range (nearest available frame)."""
    av = np.array(sorted(available))
    if len(av) == 0:
        raise SystemExit("no frame has depth (and a mask) for every camera")
    targets = np.linspace(av[0], av[-1], max(1, n))
    out = sorted({int(av[np.argmin(np.abs(av - t))]) for t in targets})
    return out


def pixel_rays(cam: dm.CameraPose, h: int, w: int) -> np.ndarray:
    """Undistorted normalised (x/z, y/z) of the depth-grid pixel centres, [h, w, 2]."""
    sx, sy = cam.width / w, cam.height / h
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * sx - 0.5, (np.arange(h) + 0.5) * sy - 0.5)
    pix = np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float64)
    if np.any(cam.dist != 0):
        xy = cv2.undistortPoints(pix[:, None], cam.K, cam.dist)[:, 0]
    else:
        K = cam.K
        xy = np.stack([(pix[:, 0] - K[0, 2]) / K[0, 0], (pix[:, 1] - K[1, 2]) / K[1, 1]], 1)
    return xy.reshape(h, w, 2).astype(np.float64)


def project(cam: dm.CameraPose, Xw: np.ndarray):
    """Full-resolution float pixel coordinates and camera z of world points (z <= 0.05: u = -inf)."""
    Xc = Xw @ cam.R.T + cam.t
    z = Xc[:, 2]
    uv = np.full((len(Xw), 2), -1e9)
    front = z > 0.05
    if front.any():
        if np.any(cam.dist != 0):
            p, _ = cv2.projectPoints(Xc[front], np.zeros(3), np.zeros(3), cam.K, cam.dist)
            uv[front] = p[:, 0]
        else:
            uv[front, 0] = cam.K[0, 0] * Xc[front, 0] / z[front] + cam.K[0, 2]
            uv[front, 1] = cam.K[1, 1] * Xc[front, 1] / z[front] + cam.K[1, 2]
    return uv, z


def edge_safe(d: np.ndarray, radius: int, rel: float, abs_m: float, hole_erode: int) -> np.ndarray:
    """Valid depth whose (2r+1)^2 neighbourhood spans at most abs + rel * d (invalid neighbours
    ignored) and that is at least ``hole_erode`` pixels from an invalid pixel."""
    valid = d > 0
    ok = valid.copy()
    if radius > 0:
        ker = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        hi = cv2.dilate(np.where(valid, d, 0).astype(np.float32), ker)
        lo = cv2.erode(np.where(valid, d, 1e6).astype(np.float32), ker)
        ok &= (hi - lo) <= abs_m + rel * d
    if hole_erode > 0:
        ker = np.ones((2 * hole_erode + 1, 2 * hole_erode + 1), np.uint8)
        ok &= cv2.erode(valid.astype(np.uint8), ker, borderType=cv2.BORDER_REPLICATE) > 0
    return ok


def normals_cam(d: np.ndarray, rays: np.ndarray, ok: np.ndarray, step: int):
    """Unit surface normals (camera frame, facing the camera) from central differences of
    the back-projected depth over +-``step`` pixels; valid where all 4 taps are ``ok``."""
    P = np.concatenate([rays * d[..., None], d[..., None]], -1)
    h, w = d.shape
    k = step
    n = np.zeros_like(P)
    good = np.zeros((h, w), bool)
    du = P[k:-k, 2 * k:] - P[k:-k, :-2 * k]
    dv = P[2 * k:, k:-k] - P[:-2 * k, k:-k]
    c = np.cross(du, dv)
    norm = np.linalg.norm(c, axis=-1)
    g = (ok[k:-k, 2 * k:] & ok[k:-k, :-2 * k] & ok[2 * k:, k:-k] & ok[:-2 * k, k:-k] & ok[k:-k, k:-k]
         & (norm > 0))
    c = c / np.maximum(norm, 1e-12)[..., None]
    flip = np.sum(c * P[k:-k, k:-k], -1) > 0  # must point back to the camera
    c[flip] *= -1
    n[k:-k, k:-k] = c
    good[k:-k, k:-k] = g
    return n, good


def sample_depth(d: np.ndarray, ok: np.ndarray, uw: np.ndarray, vw: np.ndarray):
    """Bilinear depth at float depth-grid coordinates where all 4 neighbours are ``ok``."""
    h, w = d.shape
    x0 = np.floor(uw).astype(np.int64)
    y0 = np.floor(vw).astype(np.int64)
    inb = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0c, y0c = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    good = inb & ok[y0c, x0c] & ok[y0c, x0c + 1] & ok[y0c + 1, x0c] & ok[y0c + 1, x0c + 1]
    ax, ay = uw - x0c, vw - y0c
    val = ((1 - ax) * (1 - ay) * d[y0c, x0c] + ax * (1 - ay) * d[y0c, x0c + 1]
           + (1 - ax) * ay * d[y0c + 1, x0c] + ax * ay * d[y0c + 1, x0c + 1])
    return np.where(good, val, 0.0), good


def robust_median(r: np.ndarray, gate_wide: float, gate: float):
    """Median of r after a wide gate around 0 and a narrow gate around the first median."""
    r = r[np.abs(r) < gate_wide]
    if len(r) == 0:
        return np.nan, np.zeros(0, bool), r
    m0 = float(np.median(r))
    inl = np.abs(r - m0) < gate
    return (float(np.median(r[inl])) if inl.any() else np.nan), inl, r


# ======================================================================================
# Measurement
# ======================================================================================
class Frame:
    """Depth, selections and lifted subject points of every camera at one frame."""

    def __init__(self, a, cams, rays, frame: int, rng: np.random.Generator):
        self.d, self.ok_tgt, self.X, self.xy, self.N = [], [], [], [], []
        ker = np.ones((2 * a.mask_erode + 1, 2 * a.mask_erode + 1), np.uint8) if a.mask_erode > 0 else None
        for ci, cam in enumerate(cams):
            d = load_depth(a.depth_root, cam.folder, frame, a.depth_scale)
            safe = edge_safe(d, a.edge_radius, a.edge_rel, a.edge_abs, a.hole_erode)
            if a.masks_root is not None:
                fg = (load_mask(a.masks_root, cam.folder, frame, d.shape) >= a.mask_threshold).astype(np.uint8)
                if ker is not None:
                    fg = cv2.erode(fg, ker, borderType=cv2.BORDER_CONSTANT)
                sel = safe & (fg > 0)
            else:
                sel = safe
            nrm, ngood = normals_cam(d, rays[ci], safe, a.normal_step)
            rows, cols = np.nonzero(sel & ngood)
            if len(rows) > a.max_points:
                k = rng.choice(len(rows), a.max_points, replace=False)
                rows, cols = rows[k], cols[k]
            z = d[rows, cols].astype(np.float64)
            xy = rays[ci][rows, cols]
            Xc = np.concatenate([xy * z[:, None], z[:, None]], 1)
            self.d.append(d)
            self.ok_tgt.append(sel)
            self.X.append((Xc - cam.t) @ cam.R)
            self.xy.append(xy)
            self.N.append(nrm[rows, cols] @ cam.R)  # world normals


def pair_residuals(a, cams, fr: Frame, i: int, j: int):
    """Residuals z_j(point of i) - d_j and coefficients c = d z_j / d z_i for one pair."""
    cj = cams[j]
    d, ok = fr.d[j], fr.ok_tgt[j]
    h, w = d.shape
    uv, zj = project(cj, fr.X[i])
    uw = (uv[:, 0] + 0.5) * w / cj.width - 0.5
    vw = (uv[:, 1] + 0.5) * h / cj.height - 0.5
    dj, good = sample_depth(d, ok, uw, vw)
    good &= zj > 0.05
    N = fr.N[i]
    if a.max_view_angle > 0:
        # the surface must face both cameras (no far-side or grazing matches)
        cos_max = np.cos(np.deg2rad(a.max_view_angle))
        for cam in (cams[i], cj):
            v = cam.center[None] - fr.X[i]
            good &= np.sum(N * v, 1) > cos_max * np.linalg.norm(v, axis=1)
    # Sensitivity of the residual to camera i's offset.  Moving the point by o_i along
    # camera i's ray (unit z in i) moves it off the surface; camera j looks the surface up
    # along its own ray (unit z in j), so the residual changes by o_i (n.ray_i) / (n.ray_j).
    # (Not the z-component of ray_i in j: the lookup point slides along the surface.)
    ray_i = np.concatenate([fr.xy[i][good], np.ones((int(good.sum()), 1))], 1) @ cams[i].R
    ray_j = (fr.X[i][good] - cj.center[None]) / zj[good][:, None]
    n_i, n_j = np.sum(N[good] * ray_i, 1), np.sum(N[good] * ray_j, 1)
    c = n_i / np.where(np.abs(n_j) > 1e-6, n_j, np.nan)
    keep = np.isfinite(c)
    return (zj[good] - dj[good])[keep], c[keep]


def measure(a, cams, frames: list[int], log=print):
    """Per frame and ordered pair: (median residual m, mean coefficient c, inliers n)."""
    n = len(cams)
    rays = [pixel_rays(cam, *load_depth(a.depth_root, cam.folder, frames[0], a.depth_scale).shape) for cam in cams]
    rng = np.random.default_rng(a.seed)
    meas = {}  # (frame, i, j) -> (m, c, n)
    for f in frames:
        t0 = time.perf_counter()
        fr = Frame(a, cams, rays, f, rng)
        cnt = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r, c = pair_residuals(a, cams, fr, i, j)
                med, inl, rg = robust_median(r, a.gate_wide, a.gate)
                if inl.sum() >= a.min_points and np.isfinite(med):
                    cg = c[np.abs(r) < a.gate_wide][inl]
                    meas[(f, i, j)] = (med, float(cg.mean()), int(inl.sum()))
                    cnt += 1
        log(f"  frame {f:6d}: {sum(len(x) for x in fr.X):,} subject points, {cnt} pairs measured "
            f"({time.perf_counter() - t0:.1f} s)")
    return meas


# ======================================================================================
# Solve
# ======================================================================================
def solve(eqs, n: int, gauge: str, huber: float, iters: int):
    """eqs: list of (i, j, r, c, weight).  Minimise sum w * rho(r + c o_i - o_j) over o.

    The common shift is part of the fit (it is what the c_ij != 1 geometry says, and it
    absorbs a rig-scale error that looks the same), so the differences are never distorted
    by forcing it to zero.  gauge 'free' returns that solution; 'mean' removes its mean.
    (A rank-deficient system, all c = 1, gets the minimum-norm, i.e. mean-zero, solution.)
    Returns (o [n], residuals of the returned o [m], robust weights [m])."""
    m = len(eqs)
    A = np.zeros((m, n))
    b = np.zeros(m)
    w0 = np.ones(m)
    for k, (i, j, r, c, w) in enumerate(eqs):
        A[k, i] = c
        A[k, j] = -1.0
        b[k] = -r
        w0[k] = w
    rw = np.ones(m)
    o = np.zeros(n)
    for _ in range(max(1, iters + 1)):
        W = w0 * rw
        AtA = A.T @ (W[:, None] * A)
        Atb = A.T @ (W * b)
        o = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
        res = A @ o - b
        if huber <= 0 or iters <= 0:
            break
        rw = np.minimum(1.0, huber / np.maximum(np.abs(res), 1e-12))
    if gauge == "mean":
        o = o - o.mean()
    return o, A @ o - b, rw


def free_conditioning(eqs, n: int) -> float:
    """Ratio of the smallest to the largest singular value of the unconstrained system."""
    A = np.zeros((len(eqs), n))
    for k, (i, j, _r, c, _w) in enumerate(eqs):
        A[k, i] = c
        A[k, j] = -1.0
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[-1] / s[0]) if s[0] > 0 else 0.0


def pair_equations(meas, frames, n, min_frames: int):
    """Aggregate per-frame pair medians: median over frames, mean c, total inliers."""
    eqs, info = [], {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rows = [(f, *meas[(f, i, j)]) for f in frames if (f, i, j) in meas]
            if len(rows) < min_frames:
                continue
            ms = np.array([r[1] for r in rows])
            cs = np.array([r[2] for r in rows])
            ns = np.array([r[3] for r in rows])
            r = float(np.median(ms))
            eqs.append((i, j, r, float(np.average(cs, weights=ns)), 1.0))
            info[(i, j)] = {"frames": [int(r[0]) for r in rows], "frame_medians": ms, "n": int(ns.sum()),
                            "c": float(np.average(cs, weights=ns))}
    return eqs, info


# ======================================================================================
# Gauge from colour-triangulated points
# ======================================================================================
def load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        P = np.load(path).astype(np.float64)
        return P.reshape(-1, 3)
    if path.suffix.lower() == ".bin":
        raise SystemExit(f"{path}: binary COLMAP models are not read; pass points3D.txt or an .npy")
    pts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = line.split()
        pts.append([float(v[1]), float(v[2]), float(v[3])])
    return np.array(pts, np.float64).reshape(-1, 3)


def points_offsets(a, cams, frames, P: np.ndarray):
    """Per camera: robust median over frames of median(camera z of point - measured depth)
    at edge-safe pixels (no mask: calibration points are mostly static background)."""
    out = {}
    for ci, cam in enumerate(cams):
        per = []
        zs = []
        for f in frames:
            d = load_depth(a.depth_root, cam.folder, f, a.depth_scale)
            ok = edge_safe(d, a.edge_radius, a.edge_rel, a.edge_abs, a.hole_erode)
            h, w = d.shape
            uv, z = project(cam, P)
            uw = (uv[:, 0] + 0.5) * w / cam.width - 0.5
            vw = (uv[:, 1] + 0.5) * h / cam.height - 0.5
            dm_, good = sample_depth(d, ok, uw, vw)
            good &= z > 0.05
            if a.gauge_max_depth > 0:
                good &= z <= a.gauge_max_depth
            r = z[good] - dm_[good]
            med, inl, rg = robust_median(r, a.gate_wide, a.gate)
            if np.isfinite(med) and inl.sum() >= a.gauge_min_points:
                per.append(med)
                zs.append(float(np.median(z[good][np.abs(r) < a.gate_wide][inl])))
        if per:
            out[cam.folder] = {"offset_m": float(np.median(per)), "frames": len(per),
                               "frame_spread_mm": float(np.std(per) * 1000), "median_depth_m": float(np.median(zs))}
    return out


# ======================================================================================
# CLI
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("inputs")
    g.add_argument("--data-dir", default=".")
    g.add_argument("--reference-json", default="reference_export.json")
    g.add_argument("--depth-dir", default="depth_registered", help="registered z-depth (.png or .npy)")
    g.add_argument("--depth-scale", type=float, default=0.001, help="metres per PNG unit")
    g.add_argument("--masks-dir", default="masks_refined",
                   help="subject masks (255 = subject); 'none' uses all edge-safe depth (not recommended)")
    g.add_argument("--output", default=None, help="default: <data-dir>/depth_offsets.json")
    g = p.add_argument_group("sampling")
    g.add_argument("--frames", default=None, help="comma-separated frame numbers; default: --num-frames spread")
    g.add_argument("--num-frames", type=int, default=12, help="frames spread evenly over the clip")
    g.add_argument("--max-points", type=int, default=40000, help="subject points per camera and frame")
    g.add_argument("--seed", type=int, default=0)
    g = p.add_argument_group("selection")
    g.add_argument("--mask-threshold", type=float, default=0.5)
    g.add_argument("--mask-erode", type=int, default=4, help="depth-grid pixels; keeps away from the silhouette")
    g.add_argument("--edge-radius", type=int, default=3, help="depth-grid pixels of the discontinuity window")
    g.add_argument("--edge-rel", type=float, default=0.015, help="max depth range in the window, x depth")
    g.add_argument("--edge-abs", type=float, default=0.0, help="m, added to --edge-rel x depth")
    g.add_argument("--hole-erode", type=int, default=1)
    g.add_argument("--max-view-angle", type=float, default=60.0,
                   help="deg; the surface normal must be within this angle of both cameras' view rays (0 = off)")
    g.add_argument("--normal-step", type=int, default=2, help="depth-grid pixels of the normal's central differences")
    g.add_argument("--gate-wide", type=float, default=0.06, help="m; first gate around 0")
    g.add_argument("--gate", type=float, default=0.025, help="m; second gate around the first median")
    g.add_argument("--min-points", type=int, default=300, help="residuals per pair and frame")
    g.add_argument("--min-frames", type=int, default=2, help="frames per pair")
    g = p.add_argument_group("solve")
    g.add_argument("--gauge", choices=["mean", "free"], default="mean",
                   help="mean: corrections sum to zero; free: unconstrained (geometry sets the common shift)")
    g.add_argument("--huber-mm", type=float, default=0.0, help="IRLS Huber threshold on pair residuals; 0 = plain LS")
    g.add_argument("--irls-iters", type=int, default=10)
    g.add_argument("--gauge-points", default=None,
                   help="colour-triangulated world points (.npy [N,3] or COLMAP points3D.txt) for the reported "
                        "common-shift check; default: k4a_report/calibration_points.npy when it exists; 'none' skips")
    g.add_argument("--gauge-max-depth", type=float, default=0.0, help="m; ignore farther gauge points (0 = all)")
    g.add_argument("--gauge-min-points", type=int, default=20)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    t_all = time.perf_counter()
    data = Path(a.data_dir)
    cams = dm.load_cameras(data, a.reference_json)
    n = len(cams)
    if n < 2:
        raise SystemExit("need at least 2 cameras")
    names = [c.folder for c in cams]
    a.depth_root = data / a.depth_dir
    a.masks_root = None if a.masks_dir in (None, "", "none", "None") else data / a.masks_dir
    available = frames_in(a.depth_root, cams)
    if a.masks_root is not None:
        available &= frames_in(a.masks_root, cams)
    if a.frames:
        frames = sorted({int(x) for x in a.frames.split(",") if x.strip()})
        missing = [f for f in frames if f not in available]
        if missing:
            raise SystemExit(f"frames without depth/mask for every camera: {missing}")
    else:
        frames = pick_frames(available, a.num_frames)
    print(f"[depth_offsets] {n} cameras, frames {frames}, depth {a.depth_dir}, masks {a.masks_dir}")

    meas = measure(a, cams, frames)
    eqs, info = pair_equations(meas, frames, n, a.min_frames)
    if not eqs:
        raise SystemExit("no camera pair has enough overlapping subject depth")
    connected = {i for e in eqs for i in e[:2]}
    if len(connected) < n:
        raise SystemExit(f"cameras without any measured pair: {[names[i] for i in range(n) if i not in connected]}")

    huber = a.huber_mm / 1000.0
    o_mean, _, rw_mean = solve(eqs, n, "mean", huber, a.irls_iters)
    o_free, res_free, rw_free = solve(eqs, n, "free", huber, a.irls_iters)
    o, rw = (o_free, rw_free) if a.gauge == "free" else (o_mean, rw_mean)
    A = np.zeros((len(eqs), n))
    b = np.array([-e[2] for e in eqs])
    for k, (i, j, _r, c, _w) in enumerate(eqs):
        A[k, i], A[k, j] = c, -1.0
    res = A @ o - b
    before = float(np.sqrt(np.mean(b ** 2)) * 1000)
    after = float(np.sqrt(np.mean(res ** 2)) * 1000)

    # time stability: the same solve on every frame alone (mean gauge)
    per_frame = {}
    for f in frames:
        fe = [(i, j, m, c, 1.0) for (ff, i, j), (m, c, _n) in meas.items() if ff == f]
        if len({x for e in fe for x in e[:2]}) == n:
            per_frame[f] = solve(fe, n, "mean", huber, a.irls_iters)[0]
    spread = {}
    if per_frame:
        Mf = np.stack(list(per_frame.values()))
        spread = {names[k]: float(Mf[:, k].std() * 1000) for k in range(n)}

    # independent check of the common shift from colour-triangulated points (reported only)
    gauge_points = None
    if a.gauge_points not in ("none", "None", ""):
        gpath = Path(a.gauge_points) if a.gauge_points else data / "k4a_report" / "calibration_points.npy"
        if not gpath.is_absolute():
            gpath = gpath if gpath.exists() else data / gpath
        if gpath.exists():
            P = load_points(gpath)
            po = points_offsets(a, cams, frames, P) if len(P) else {}
            if po:
                diffs = np.array([po[c]["offset_m"] - o_mean[names.index(c)] for c in po])
                gauge_points = {
                    "path": str(gpath), "points": int(len(P)), "cameras": len(po),
                    "common_shift_mm": round(float(np.median(diffs)) * 1000, 2),
                    "shift_spread_over_cameras_mm": round(float(np.std(diffs)) * 1000, 2),
                    "per_camera_mm": {c: {"offset_mm": round(v["offset_m"] * 1000, 2), "frames": v["frames"],
                                          "frame_spread_mm": round(v["frame_spread_mm"], 2),
                                          "median_depth_m": round(v["median_depth_m"], 3)} for c, v in po.items()},
                }
        elif a.gauge_points:
            raise SystemExit(f"--gauge-points: {gpath} not found")

    per_pair = {}
    for k, (i, j, r, c, _w) in enumerate(eqs):
        inf = info[(i, j)]
        per_pair[f"{names[i]}->{names[j]}"] = {
            "points": inf["n"], "frames": len(inf["frames"]), "coef": round(inf["c"], 4),
            "before_mm": round(r * 1000, 2), "after_mm": round(float(res[k]) * 1000, 2),
            "irls_weight": round(float(rw[k]), 3),
            "frame_medians_mm": {str(f): round(float(m) * 1000, 2) for f, m in zip(inf["frames"], inf["frame_medians"])},
            "frame_spread_mm": round(float(np.std(inf["frame_medians"]) * 1000), 2),
        }

    out = {
        "contract": CONTRACT,
        "offsets_m": {names[k]: round(float(o[k]), 6) for k in range(n)},
        "gauge": a.gauge,
        "frames": frames,
        "pair_residual_rms_mm": {"before": round(before, 2), "after": round(after, 2)},
        "per_pair": per_pair,
        "per_frame_offsets_mm": {str(f): {names[k]: round(float(v[k]) * 1000, 2) for k in range(n)}
                                 for f, v in per_frame.items()},
        "per_frame_spread_mm": {k: round(v, 2) for k, v in spread.items()},
        "free_solution": {
            "offsets_mm": {names[k]: round(float(o_free[k]) * 1000, 2) for k in range(n)},
            "common_shift_mm": round(float(o_free.mean()) * 1000, 2),
            "pair_residual_rms_mm": round(float(np.sqrt(np.mean(res_free ** 2)) * 1000), 2),
            "conditioning": round(free_conditioning(eqs, n), 4),
            "note": "common level implied by the calibrated geometry alone; confounded with the rig scale",
        },
        "gauge_points": gauge_points,
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
    }
    dst = Path(a.output) if a.output else data / "depth_offsets.json"
    if a.output and not dst.is_absolute():
        dst = data / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    tmp.replace(dst)

    print(f"[depth_offsets] {len(eqs)} pair equations; pair residual RMS {before:.1f} -> {after:.1f} mm "
          f"(gauge {a.gauge})")
    print("  offsets to ADD (mm): " + "  ".join(f"{names[k]} {o[k] * 1000:+.1f}" for k in range(n)))
    if spread:
        print("  per-frame spread (std over frames, mm): " + "  ".join(f"{k} {v:.1f}" for k, v in spread.items()))
    print(f"  free solution: common shift {o_free.mean() * 1000:+.1f} mm (conditioning "
          f"{out['free_solution']['conditioning']:.3f}, RMS {out['free_solution']['pair_residual_rms_mm']:.1f} mm)")
    if gauge_points is not None:
        print(f"  gauge points ({gauge_points['points']} pts, {gauge_points['cameras']} cameras): common shift "
              f"{gauge_points['common_shift_mm']:+.1f} mm, spread over cameras "
              f"{gauge_points['shift_spread_over_cameras_mm']:.1f} mm (reported only)")
    print(f"[depth_offsets] wrote {dst} ({time.perf_counter() - t_all:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
