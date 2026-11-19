#!/usr/bin/env python3
"""Foreground masks from multi-view metric depth - any depth-capable capture rig.

Inputs (capture data directory)::

    reference_export.json      camera intrinsics + world->camera poses (prep convention;
                               PINHOLE / OPENCV / FULL_OPENCV / RADIAL ... models)
    images/<cam>/<frame>.jpg   only for the QA sheet
    + one depth source:

    --depth-source registered  (generic contract, any sensor)
        depth_registered/<cam>/<frame:06d>.png   uint16 depth, ``--depth-scale`` metres per
                               unit (default 0.001 = millimetres), 0 = invalid; z-depth
                               along the optical axis of that camera's *colour* image.
                               Any resolution with the image's aspect ratio (intrinsics are
                               scaled). ``.npy`` float32 metres is also accepted.
    --depth-source k4a          Azure Kinect raw depth written by extract_azure_kinect.py
                               (depth/<cam>/<frame>.png + k4a_cameras.json); no registration
                               needed.
    --depth-source auto         k4a when k4a_cameras.json exists, else registered.

No world frame is assumed.  The floor plane and the up direction are estimated from the
fused depth (cameras are assumed roughly upright: image "up" points away from the floor),
unless ``--no-floor`` is given.

Per frame:

1. Every camera's depth is unprojected and moved to world space with its pose.
2. Foreground = points inside a **stage cylinder** standing on the floor: horizontal
   distance to the stage centre < radius, floor-margin < height < top.  Low points
   (< ``--low-band``) only count where the same floor column also has foreground above them
   (keeps feet / chair legs, drops cables and floor noise).  Only the main 3D-connected
   cluster survives (stray clumps are dropped).
3. The fused foreground of **all** cameras is projected into each camera, so a hole in one
   sensor's depth is filled by the others; a point is dropped where that camera's own depth
   shows background clearly in front of it (occluders).
4. Close, bounded hole fill, speck removal, safety dilation and a feathered edge give an
   8-bit alpha mask (255 = subject, 0 = background).

Outputs: ``masks/<cam>/<frame:06d>.png`` (pipeline mask layout, consumed by
``prep-roma-trigs-memory.py --masks-root`` and ``trigs-train-colmap --masks-root``),
``<report-dir>/masks.json`` and ``<report-dir>/mask_sheet.jpg``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ======================================================================================
# Cameras
# ======================================================================================
@dataclass
class CameraPose:
    folder: str
    K: np.ndarray  # 3x3, image pixels
    dist: np.ndarray  # OpenCV distortion vector (zeros for pinhole)
    width: int
    height: int
    R: np.ndarray  # world -> camera
    t: np.ndarray
    model: str = "PINHOLE"

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


def _intrinsics(model: str, params: list[float]) -> tuple[np.ndarray, np.ndarray]:
    p = [float(v) for v in params]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = p[0], p[1], p[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        k = p[3:]
        dist = np.array([k[0] if k else 0.0, k[1] if len(k) > 1 else 0.0, 0.0, 0.0])
    elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])
        dist = np.array(p[4:] if len(p) > 4 else [0.0, 0.0, 0.0, 0.0])
    else:
        raise ValueError(f"camera model {model} is not supported by depth_masks (fisheye: undistort first)")
    return K, dist.astype(np.float64)


def load_cameras(data_dir: Path, reference_json: str = "reference_export.json") -> list[CameraPose]:
    ref = json.loads((Path(data_dir) / reference_json).read_text(encoding="utf-8"))
    cams = []
    for c in sorted(ref["cameras"], key=lambda c: c["folder"]):
        K, dist = _intrinsics(c["camera_model_name"], c["params"])
        cams.append(CameraPose(c["folder"], K, dist, int(c["width"]), int(c["height"]),
                               np.asarray(c["R"], dtype=np.float64), np.asarray(c["t"], dtype=np.float64),
                               c["camera_model_name"]))
    return cams


def project(cam: CameraPose, Xw: np.ndarray):
    """Pixel coordinates (int) and depth of world points inside the image."""
    Xc = (Xw @ cam.R.T + cam.t).astype(np.float64)
    z = Xc[:, 2]
    ok = z > 0.05
    Xc, z = Xc[ok], z[ok]
    if np.any(cam.dist != 0):
        uv, _ = cv2.projectPoints(Xc, np.zeros(3), np.zeros(3), cam.K, cam.dist)
        uv = uv[:, 0]
    else:
        uv = np.c_[cam.K[0, 0] * Xc[:, 0] / z + cam.K[0, 2], cam.K[1, 1] * Xc[:, 1] / z + cam.K[1, 2]]
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    inb = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    return u[inb], v[inb], z[inb]


def to_world(cam: CameraPose, Xc: np.ndarray) -> np.ndarray:
    return ((Xc - cam.t) @ cam.R).astype(np.float32)


# ======================================================================================
# Depth sources: points_cam(camera_index, frame) -> [N,3] float32 metres, colour-camera frame
# ======================================================================================
class RegisteredDepthSource:
    """Generic: depth maps registered to each camera's colour image."""

    name = "registered"

    def __init__(self, data_dir: Path, cams: list[CameraPose], depth_dir: str = "depth_registered",
                 depth_scale: float = 0.001, pattern: str = "{frame:06d}"):
        self.root = Path(data_dir) / depth_dir
        self.cams = cams
        self.scale = depth_scale
        self.pattern = pattern
        self._rays: dict[tuple[int, int, int], np.ndarray] = {}
        if not self.root.is_dir():
            raise FileNotFoundError(f"registered depth directory not found: {self.root}")

    def _load(self, ci: int, frame: int) -> np.ndarray:
        base = self.root / self.cams[ci].folder / self.pattern.format(frame=frame)
        if base.with_suffix(".npy").exists():
            return np.load(base.with_suffix(".npy")).astype(np.float32)
        img = cv2.imread(str(base.with_suffix(".png")), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"missing depth {base}.png/.npy")
        if img.ndim == 3:
            img = img[..., 0]
        return img.astype(np.float32) * self.scale

    def _rays_for(self, ci: int, h: int, w: int) -> np.ndarray:
        key = (ci, h, w)
        if key not in self._rays:
            cam = self.cams[ci]
            if abs(w / h - cam.width / cam.height) > 0.01:
                raise ValueError(f"{cam.folder}: depth {w}x{h} does not match image aspect {cam.width}x{cam.height}")
            sx, sy = cam.width / w, cam.height / h
            uu, vv = np.meshgrid((np.arange(w) + 0.5) * sx - 0.5, (np.arange(h) + 0.5) * sy - 0.5)
            pix = np.stack([uu.ravel(), vv.ravel()], 1)[:, None].astype(np.float64)
            self._rays[key] = cv2.undistortPoints(pix, cam.K, cam.dist)[:, 0].reshape(h, w, 2).astype(np.float32)
        return self._rays[key]

    def points_cam(self, ci: int, frame: int) -> np.ndarray:
        d = self._load(ci, frame)
        rays = self._rays_for(ci, *d.shape[:2])
        valid = np.isfinite(d) & (d > 0)
        z = d[valid]
        return np.concatenate([rays[valid] * z[:, None], z[:, None]], axis=1).astype(np.float32)


class K4ARawDepthSource:
    """Azure Kinect raw depth images + factory depth calibration (extract_azure_kinect.py)."""

    name = "k4a"

    def __init__(self, data_dir: Path, cams: list[CameraPose], depth_dir: str = "depth"):
        meta = json.loads((Path(data_dir) / "k4a_cameras.json").read_text(encoding="utf-8"))["cameras"]
        self.root = Path(data_dir) / depth_dir
        self.cams = cams
        self.calib = []
        for cam in cams:
            c = meta[cam.folder]
            di = c.get("depth_intrinsics")
            if di is None:
                raise ValueError(f"{cam.folder}: recording has no depth stream")
            Kd = np.array([[di["fx"], 0, di["cx"]], [0, di["fy"], di["cy"]], [0, 0, 1.0]])
            uu, vv = np.meshgrid(np.arange(di["width"]), np.arange(di["height"]))
            pix = np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float64)[:, None]
            rays = cv2.undistortPoints(pix, Kd, np.asarray(di["dist_opencv_rational"], dtype=np.float64))
            self.calib.append((rays[:, 0].reshape(di["height"], di["width"], 2).astype(np.float32),
                               np.asarray(c["depth_to_color_R"], dtype=np.float32),
                               np.asarray(c["depth_to_color_t"], dtype=np.float32)))

    def points_cam(self, ci: int, frame: int) -> np.ndarray:
        rays, R, t = self.calib[ci]
        d = cv2.imread(str(self.root / self.cams[ci].folder / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(f"missing depth for camera {self.cams[ci].folder} frame {frame}")
        valid = d > 0
        z = d[valid].astype(np.float32) / 1000.0
        Xd = np.concatenate([rays[valid] * z[:, None], z[:, None]], axis=1)
        return Xd @ R.T + t


def make_source(kind: str, data_dir: Path, cams: list[CameraPose], depth_dir: str | None, depth_scale: float):
    data_dir = Path(data_dir)
    if kind == "auto":
        kind = "k4a" if (data_dir / "k4a_cameras.json").exists() else "registered"
    if kind == "k4a":
        return K4ARawDepthSource(data_dir, cams, depth_dir or "depth")
    return RegisteredDepthSource(data_dir, cams, depth_dir or "depth_registered", depth_scale)


# ======================================================================================
# Ground frame (floor plane + up), estimated - no world convention assumed
# ======================================================================================
@dataclass
class Ground:
    up: np.ndarray  # unit vector, world
    d: float  # height(X) = up . X + d  (floor at 0)
    e1: np.ndarray  # horizontal basis
    e2: np.ndarray
    has_floor: bool = True

    def height(self, X: np.ndarray) -> np.ndarray:
        return X @ self.up.astype(np.float32) + np.float32(self.d)

    def plane_xy(self, X: np.ndarray) -> np.ndarray:
        return np.stack([X @ self.e1.astype(np.float32), X @ self.e2.astype(np.float32)], axis=1)

    def to_dict(self) -> dict:
        return {"up": self.up.tolist(), "d": self.d, "e1": self.e1.tolist(), "e2": self.e2.tolist(),
                "has_floor": self.has_floor}


def _basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 0, 1.0])
    e1 = a - (a @ up) * up
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def camera_up_hint(cams: list[CameraPose]) -> np.ndarray:
    """Mean image-up direction (camera -y axis) in world: assumes roughly upright images."""
    v = -np.mean([c.R.T @ np.array([0, 1.0, 0]) for c in cams], axis=0)
    return v / np.linalg.norm(v)


def fit_floor(points: np.ndarray, up_hint: np.ndarray, cam_centers: np.ndarray, iters: int = 3000,
              thresh: float = 0.02, max_tilt_deg: float = 25.0, min_cam_height: float = 0.3, seed: int = 0):
    """RANSAC plane with normal within ``max_tilt_deg`` of ``up_hint``, below every camera,
    maximising support.  Returns (normal, d, support) with normal . X + d = height, or None."""
    rng = np.random.default_rng(seed)
    n = len(points)
    if n < 3:
        return None
    best = (0, None)
    cos_max = math.cos(math.radians(max_tilt_deg))
    for _ in range(iters):
        s = points[rng.choice(n, 3, replace=False)].astype(np.float64)
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        L = np.linalg.norm(nrm)
        if L < 1e-9:
            continue
        nrm /= L
        if nrm @ up_hint < 0:
            nrm = -nrm
        if nrm @ up_hint < cos_max:
            continue
        d = -nrm @ s[0]
        if np.any(cam_centers @ nrm + d < min_cam_height):
            continue
        support = int((np.abs(points @ nrm + d) < thresh).sum())
        if support > best[0]:
            best = (support, (nrm, d))
    if best[1] is None:
        return None
    nrm, d = best[1]
    inl = points[np.abs(points @ nrm + d) < thresh].astype(np.float64)
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    nrm = vt[2] if vt[2] @ up_hint > 0 else -vt[2]
    return nrm, float(-nrm @ c), len(inl)


def estimate_ground(cams, source, frames, no_floor: bool = False, max_points: int = 300_000) -> Ground:
    up_hint = camera_up_hint(cams)
    if no_floor:
        e1, e2 = _basis(up_hint)
        return Ground(up_hint, 0.0, e1, e2, has_floor=False)
    pts = []
    for f in frames:
        for ci, cam in enumerate(cams):
            Xw = to_world(cam, source.points_cam(ci, f))
            pts.append(Xw[:: max(1, len(Xw) // 20000)])
    P = np.concatenate(pts)
    if len(P) > max_points:
        P = P[np.random.default_rng(1).choice(len(P), max_points, replace=False)]
    centers = np.array([c.center for c in cams])
    fl = fit_floor(P, up_hint, centers)
    if fl is None:
        raise SystemExit("floor plane not found (pass --no-floor for scenes without a visible floor)")
    up, d, _ = fl
    e1, e2 = _basis(up)
    return Ground(up, d, e1, e2, has_floor=True)


def look_at_point(cams: list[CameraPose]) -> np.ndarray:
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c in cams:
        d = c.R.T @ np.array([0, 0, 1.0])
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c.center
    return np.linalg.solve(A, b)


# ======================================================================================
# Mask construction
# ======================================================================================
@dataclass
class Params:
    center: tuple[float, float]  # stage centre in ground-plane coordinates (e1, e2)
    radius: float
    floor_margin: float
    top: float
    occlusion_margin: float = 0.15
    splat_px: int = 3
    close_px: int = 9
    dilate_px: int = 6
    feather_px: float = 2.0
    min_component_frac: float = 0.002
    max_hole_frac: float = 0.003
    low_band: float = 0.12
    column_cell: float = 0.04
    column_min_points: int = 5
    cluster_voxel: float = 0.05
    cluster_keep_ratio: float = 0.15
    floor_bias: tuple[float, ...] = field(default_factory=tuple)  # per-camera height offsets (m)


def _disk(r: int) -> np.ndarray:
    r = max(int(r), 0)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def fill_holes(binary: np.ndarray, max_area: float) -> np.ndarray:
    """Fill enclosed background regions up to ``max_area`` pixels (depth dropouts); larger
    enclosed regions such as floor seen between chair legs stay background."""
    b = (binary > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - b, connectivity=4)
    h, w = b.shape
    fill = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        fill[i] = (not touches) and area <= max_area
    return np.where(fill[lab], 1, b).astype(np.uint8)


def main_cluster(points: np.ndarray, voxel: float, keep_ratio: float) -> np.ndarray:
    """Points in the largest 26-connected voxel cluster plus clusters with >= keep_ratio of it."""
    if len(points) == 0:
        return np.zeros(0, bool)
    from scipy import ndimage

    ijk = np.floor((points - points.min(axis=0)) / voxel).astype(np.int32)
    occ = np.zeros(ijk.max(axis=0) + 1, np.int32)
    np.add.at(occ, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1)
    lab, n = ndimage.label(occ > 0, structure=np.ones((3, 3, 3), bool))
    if n <= 1:
        return np.ones(len(points), bool)
    counts = ndimage.sum(occ, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = counts >= keep_ratio * counts.max()
    return keep[lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]]]


def foreground_select(xy: np.ndarray, h: np.ndarray, p: Params) -> np.ndarray:
    dx = xy[:, 0] - p.center[0]
    dy = xy[:, 1] - p.center[1]
    return (dx * dx + dy * dy < p.radius ** 2) & (h > p.floor_margin) & (h < p.top)


def column_support(xys, hs, fg, p: Params):
    """Low foreground points need foreground above ``low_band`` in the same floor column."""
    if p.low_band <= p.floor_margin:
        return fg
    lo = np.array([p.center[0] - p.radius, p.center[1] - p.radius], np.float32)
    n = int(np.ceil(2 * p.radius / p.column_cell)) + 1

    def cells(xy):
        return np.clip(np.floor((xy - lo) / p.column_cell).astype(np.int32), 0, n - 1)

    count = np.zeros((n, n), np.int32)
    for xy, h, f in zip(xys, hs, fg):
        sel = f & (h >= p.low_band)
        if sel.any():
            ij = cells(xy[sel])
            np.add.at(count, (ij[:, 1], ij[:, 0]), 1)
    supported = cv2.dilate((count >= p.column_min_points).astype(np.uint8), _disk(1)) > 0
    out = []
    for xy, h, f in zip(xys, hs, fg):
        low = f & (h < p.low_band)
        if low.any():
            ij = cells(xy[low])
            f = f.copy()
            f[np.flatnonzero(low)[~supported[ij[:, 1], ij[:, 0]]]] = False
        out.append(f)
    return out


def build_masks(cams, world_pts: list[np.ndarray], ground: Ground, p: Params):
    """world_pts[i]: [N,3] world points from camera i's depth. Returns (uint8 masks, stats)."""
    xys, hs, fg = [], [], []
    for ci, Xw in enumerate(world_pts):
        h = ground.height(Xw) if ground.has_floor else np.full(len(Xw), np.float32(1.0))
        if p.floor_bias:
            h = h - np.float32(p.floor_bias[ci])
        xy = ground.plane_xy(Xw)
        xys.append(xy)
        hs.append(h)
        fg.append(foreground_select(xy, h, p))
    if ground.has_floor:
        fg = column_support(xys, hs, fg, p)
    fused = [w[f] for w, f in zip(world_pts, fg)]
    fused = np.concatenate(fused) if fused else np.zeros((0, 3), np.float32)
    if p.cluster_voxel > 0 and len(fused):
        fused = fused[main_cluster(fused, p.cluster_voxel, p.cluster_keep_ratio)]

    masks, stats = [], []
    for ci, cam in enumerate(cams):
        bu, bv, bz = project(cam, world_pts[ci][~fg[ci]])
        bg = np.full((cam.height, cam.width), np.inf, np.float32)
        np.minimum.at(bg, (bv, bu), bz.astype(np.float32))
        bg = cv2.erode(bg, _disk(1))
        u, v, z = project(cam, fused)
        visible = ~(bg[v, u] + p.occlusion_margin < z)
        m = np.zeros((cam.height, cam.width), np.uint8)
        m[v[visible], u[visible]] = 1
        m = cv2.dilate(m, _disk(p.splat_px))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(p.close_px))
        m = fill_holes(m, p.max_hole_frac * cam.width * cam.height)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= p.min_component_frac * cam.width * cam.height
        m = keep[lab].astype(np.uint8)
        if p.dilate_px > 0:
            m = cv2.dilate(m, _disk(p.dilate_px))
        alpha = m.astype(np.float32)
        if p.feather_px > 0:
            alpha = np.maximum(cv2.GaussianBlur(alpha, (0, 0), p.feather_px), m * 0.5)
        out = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
        masks.append(out)
        stats.append({"fg_fraction": float((out > 127).mean()), "own_fg_points": int(fg[ci].sum()),
                      "fused_points": int(len(fused))})
    return masks, stats


# ======================================================================================
# Stage estimation
# ======================================================================================
def estimate_stage(cams, source, frames, ground: Ground, floor_margin: float, search_radius: float,
                   cell: float = 0.05):
    """Subject centre (ground-plane coords) and horizontal extent: densest occupancy blob of
    above-floor points near the cameras' look-at point."""
    c0 = ground.plane_xy(look_at_point(cams)[None].astype(np.float32))[0].astype(np.float64)
    pts = []
    for f in frames:
        for ci, cam in enumerate(cams):
            Xw = to_world(cam, source.points_cam(ci, f))
            h = ground.height(Xw) if ground.has_floor else np.ones(len(Xw), np.float32)
            xy = ground.plane_xy(Xw)
            sel = (h > floor_margin) & (np.hypot(xy[:, 0] - c0[0], xy[:, 1] - c0[1]) < search_radius)
            pts.append(xy[sel][::4])
    P = np.concatenate(pts) if pts else np.zeros((0, 2))
    if len(P) < 1000:
        raise SystemExit("too few points near the stage centre; pass --center/--radius")
    n = int(np.ceil(2 * search_radius / cell)) + 1
    ij = np.clip(np.floor((P - (c0 - search_radius)) / cell).astype(int), 0, n - 1)
    occ = np.zeros((n, n), np.float32)
    np.add.at(occ, (ij[:, 1], ij[:, 0]), 1)
    occ_b = (occ > max(20, np.percentile(occ[occ > 0], 50))).astype(np.uint8)
    occ_b = cv2.morphologyEx(occ_b, cv2.MORPH_CLOSE, _disk(2))
    k, lab, _, cent = cv2.connectedComponentsWithStats(occ_b)
    if k < 2:
        raise SystemExit("could not find the subject; pass --center/--radius")
    centre_cell = np.full(2, search_radius / cell)
    best = max(range(1, k), key=lambda i: occ[lab == i].sum() / (1 + 0.1 * np.hypot(*(cent[i] - centre_cell))))
    ys, xs = np.nonzero(lab == best)
    wx = xs * cell + (c0[0] - search_radius) + cell / 2
    wy = ys * cell + (c0[1] - search_radius) + cell / 2
    w = occ[ys, xs]
    cx, cy = float(np.average(wx, weights=w)), float(np.average(wy, weights=w))
    return (cx, cy), float(np.max(np.hypot(wx - cx, wy - cy)) + cell)


def floor_bias(cams, source, frames, ground: Ground, center, inner: float, outer: float) -> list[float]:
    """Per-camera floor height on a ring around the stage (depth sensors disagree by ~1-2 cm)."""
    out = []
    for ci, cam in enumerate(cams):
        hs = []
        for f in frames:
            Xw = to_world(cam, source.points_cam(ci, f))
            xy = ground.plane_xy(Xw)
            h = ground.height(Xw)
            r = np.hypot(xy[:, 0] - center[0], xy[:, 1] - center[1])
            hs.append(h[(r > inner) & (r < outer) & (np.abs(h) < 0.08)])
        h = np.concatenate(hs)
        out.append(float(np.median(h)) if len(h) > 500 else 0.0)
    return out


# ======================================================================================
# Workers / QA / CLI
# ======================================================================================
_W: dict = {}


def _init(data: str, ref_json: str, kind: str, depth_dir, depth_scale: float, ground: dict, params: dict, out: str):
    cams = load_cameras(Path(data), ref_json)
    g = Ground(np.asarray(ground["up"]), ground["d"], np.asarray(ground["e1"]), np.asarray(ground["e2"]),
               ground["has_floor"])
    prm = dict(params)
    prm["center"] = tuple(prm["center"])
    prm["floor_bias"] = tuple(prm["floor_bias"])
    _W.update(cams=cams, source=make_source(kind, Path(data), cams, depth_dir, depth_scale), ground=g,
              params=Params(**prm), out=Path(out))


def _do_frame(f: int):
    cams, src = _W["cams"], _W["source"]
    world = [to_world(c, src.points_cam(i, f)) for i, c in enumerate(cams)]
    masks, stats = build_masks(cams, world, _W["ground"], _W["params"])
    for c, m in zip(cams, masks):
        cv2.imwrite(str(_W["out"] / c.folder / f"{f:06d}.png"), m)
    return f, stats


def qa_sheet(cams, images_root: Path, masks_root: Path, frames, path: Path, ext: str = "jpg"):
    rows = []
    for f in frames:
        tiles = []
        for c in cams:
            img = cv2.imread(str(images_root / c.folder / f"{f:06d}.{ext}"))
            m = cv2.imread(str(masks_root / c.folder / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None or m is None:
                continue
            a = (m.astype(np.float32) / 255.0)[..., None]
            dim = (img * 0.25).astype(np.uint8)
            dim[..., 2] = np.clip(dim[..., 2].astype(int) + 60, 0, 255)
            comp = (img * a + dim * (1 - a)).astype(np.uint8)
            cnts, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(comp, cnts, -1, (0, 255, 0), 3)
            cv2.putText(comp, f"{c.folder} f{f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            s = 480.0 / max(c.width, c.height)
            tiles.append(cv2.resize(comp, (int(c.width * s), int(c.height * s)), interpolation=cv2.INTER_AREA))
        if tiles:
            h = max(t.shape[0] for t in tiles)
            rows.append(np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]))
    if rows:
        w = max(r.shape[1] for r in rows)
        cv2.imwrite(str(path), np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT)
                                          for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def detect_frame_range(images_root: Path, cams, ext: str) -> tuple[int, int]:
    nums = None
    for c in cams:
        s = {int(p.stem) for p in (images_root / c.folder).glob(f"*.{ext}") if p.stem.isdigit()}
        nums = s if nums is None else nums & s
    if not nums:
        raise SystemExit(f"no common numbered .{ext} frames under {images_root}")
    return min(nums), max(nums)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--reference-json", default="reference_export.json")
    ap.add_argument("--depth-source", choices=("auto", "registered", "k4a"), default="auto")
    ap.add_argument("--depth-dir", default=None, help="default: depth_registered (registered) or depth (k4a)")
    ap.add_argument("--depth-scale", type=float, default=0.001, help="metres per PNG unit (registered source)")
    ap.add_argument("--images-dir", default="images")
    ap.add_argument("--image-ext", default="jpg")
    ap.add_argument("--output-dir", default=None, help="default: <data-dir>/masks")
    ap.add_argument("--report-dir", default=None, help="default: <data-dir>/mask_report")
    ap.add_argument("--frame-start", type=int, default=None, help="default: first common image frame")
    ap.add_argument("--frame-end", type=int, default=None, help="inclusive; default: last common image frame")
    ap.add_argument("--no-floor", action="store_true", help="no visible floor: cylinder is unbounded in height, "
                    "up = mean camera image-up")
    ap.add_argument("--center", type=float, nargs=2, metavar=("A", "B"),
                    help="stage centre in ground-plane coordinates (see masks.json 'ground'); default: estimated")
    ap.add_argument("--radius", type=float, default=None, help="stage radius (m); default: estimated + --radius-pad")
    ap.add_argument("--radius-pad", type=float, default=0.25)
    ap.add_argument("--search-radius", type=float, default=1.2, help="search area around the look-at point (m)")
    ap.add_argument("--floor-margin", type=float, default=0.025, help="below this height is floor (m)")
    ap.add_argument("--top", type=float, default=2.3, help="top of the stage volume above the floor (m)")
    ap.add_argument("--low-band", type=float, default=0.12, help="low points need column support (m); 0 = off")
    ap.add_argument("--column-cell", type=float, default=0.04)
    ap.add_argument("--column-min-points", type=int, default=5)
    ap.add_argument("--no-floor-bias", action="store_true", help="disable per-camera floor height correction")
    ap.add_argument("--occlusion-margin", type=float, default=0.15)
    ap.add_argument("--splat-px", type=int, default=3)
    ap.add_argument("--close-px", type=int, default=9)
    ap.add_argument("--dilate-px", type=int, default=6)
    ap.add_argument("--feather-px", type=float, default=2.0)
    ap.add_argument("--min-component", type=float, default=0.002)
    ap.add_argument("--max-hole", type=float, default=0.003)
    ap.add_argument("--cluster-voxel", type=float, default=0.05, help="0 = keep every 3D cluster")
    ap.add_argument("--cluster-keep-ratio", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--qa-only", action="store_true", help="only the QA frames (fast tuning)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data = Path(args.data_dir)
    out = Path(args.output_dir) if args.output_dir else data / "masks"
    report_dir = Path(args.report_dir) if args.report_dir else data / "mask_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    images_root = data / args.images_dir
    cams = load_cameras(data, args.reference_json)
    source = make_source(args.depth_source, data, cams, args.depth_dir, args.depth_scale)
    print(f"{len(cams)} cameras, depth source: {source.name}")

    lo, hi = detect_frame_range(images_root, cams, args.image_ext)
    f0 = lo if args.frame_start is None else args.frame_start
    f1 = hi if args.frame_end is None else args.frame_end
    sample = np.unique(np.linspace(f0, f1, 5).round().astype(int)).tolist()

    ground = estimate_ground(cams, source, sample[::2], no_floor=args.no_floor)
    if ground.has_floor:
        print(f"Floor plane: up {np.round(ground.up, 3).tolist()}, camera heights "
              f"{np.round([float(ground.up @ c.center + ground.d) for c in cams], 2).tolist()} m")
    if args.center is None or args.radius is None:
        # Only points above the low band locate the subject, so floor clutter (cables, bags, tape)
        # cannot stretch the stage; low parts of the subject are still kept by the column test.
        min_h = max(args.floor_margin, args.low_band) if ground.has_floor else args.floor_margin
        (cx, cy), r_est = estimate_stage(cams, source, sample, ground, min_h, args.search_radius)
        print(f"Estimated subject: centre ({cx:+.3f}, {cy:+.3f}), horizontal extent {r_est:.2f} m")
    center = tuple(args.center) if args.center is not None else (cx, cy)
    radius = args.radius if args.radius is not None else r_est + args.radius_pad
    bias = ()
    if ground.has_floor and not args.no_floor_bias:
        bias = tuple(floor_bias(cams, source, sample[::2], ground, center, radius + 0.1, radius + 0.8))
        print("Per-camera floor offset (cm): " + ", ".join(f"{c.folder} {b * 100:+.1f}" for c, b in zip(cams, bias)))
    top = args.top if ground.has_floor else 1e9
    floor_margin = args.floor_margin if ground.has_floor else -1e9
    p = Params(center=center, radius=radius, floor_margin=floor_margin, top=top,
               occlusion_margin=args.occlusion_margin, splat_px=args.splat_px, close_px=args.close_px,
               dilate_px=args.dilate_px, feather_px=args.feather_px, min_component_frac=args.min_component,
               max_hole_frac=args.max_hole, low_band=args.low_band, column_cell=args.column_cell,
               column_min_points=args.column_min_points, cluster_voxel=args.cluster_voxel,
               cluster_keep_ratio=args.cluster_keep_ratio, floor_bias=bias)
    print(f"Stage cylinder: centre ({center[0]:+.3f}, {center[1]:+.3f}), radius {radius:.2f} m"
          + (f", height {floor_margin:.3f}..{top:.2f} m" if ground.has_floor else ", unbounded height"))

    for c in cams:
        (out / c.folder).mkdir(parents=True, exist_ok=True)
    frames = sample if args.qa_only else list(range(f0, f1 + 1))
    t0 = time.time()
    per_cam: dict[str, list] = {c.folder: [] for c in cams}
    initargs = (str(data), args.reference_json, source.name, args.depth_dir, args.depth_scale, ground.to_dict(),
                {**asdict(p), "center": list(p.center), "floor_bias": list(p.floor_bias)}, str(out))
    with ProcessPoolExecutor(max_workers=max(1, args.workers), initializer=_init, initargs=initargs) as pool:
        for i, (f, stats) in enumerate(pool.map(_do_frame, frames, chunksize=2), 1):
            for c, s in zip(cams, stats):
                per_cam[c.folder].append((f, s["fg_fraction"]))
            if i % 50 == 0 or i == len(frames):
                print(f"  {i}/{len(frames)} frames ({time.time() - t0:.0f} s)", flush=True)

    qa_sheet(cams, images_root, out, sample, report_dir / "mask_sheet.jpg", args.image_ext)
    summary = {}
    for folder, rows in per_cam.items():
        fr = np.array([r[1] for r in rows])
        jumps = np.abs(np.diff(fr)) / np.maximum(fr[:-1], 1e-6) if len(fr) > 1 else np.zeros(0)
        flagged = [int(rows[i + 1][0]) for i in np.nonzero(jumps > 0.25)[0]]
        summary[folder] = {"fg_fraction_median": float(np.median(fr)), "fg_fraction_min": float(fr.min()),
                           "fg_fraction_max": float(fr.max()), "frames_with_jump_gt_25pct": flagged[:50]}
        print(f"  {folder}: foreground {np.median(fr) * 100:.1f}% (min {fr.min() * 100:.1f}, max {fr.max() * 100:.1f}); "
              f"{len(flagged)} frames with >25% coverage jumps")
    (report_dir / "masks.json").write_text(json.dumps({
        "depth_source": source.name, "ground": ground.to_dict(),
        "params": {**asdict(p), "center": list(p.center), "floor_bias": list(p.floor_bias)},
        "frames": frames if args.qa_only else [f0, f1], "qa_only": args.qa_only, "output_dir": str(out),
        "cameras": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Masks: {out}  QA: {report_dir / 'mask_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
