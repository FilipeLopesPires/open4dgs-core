#!/usr/bin/env python3
"""Multi-view consistency pass over per-camera foreground masks (opt-in post-pass).

Per-camera refinement (``refine_masks.py``) decides every camera on its own, so a part that
one camera drops (dark hair against a dark wall, a black chair seat, a laptop corner) is
missing from that camera's mask although the other cameras see it in their masks and
measure it with depth. The trainer then learns it as transparent from that view. This pass
uses all cameras of a frame at once:

1. **Source points.** For every camera, the registered depth inside its (eroded) mask is
   back-projected to world points. Only *edge-safe* depth is used (the depth range in a
   3x3 window is below ``--edge-tol``: no flying pixels at silhouettes) and only points in
   the subject volume (above the floor by ``--floor-margin``, inside the stage cylinder +
   ``--stage-margin`` when ``mask_report/masks.json`` has one).
2. **Support.** The points of every other camera are projected into camera j at depth
   resolution and splatted (closing of ``--splat-px``). A projected point *supports* pixel
   j when it agrees with camera j's own depth there (within ``--depth-tol``) or camera j
   has no depth there (ToF dropouts on dark hair, black fabric). A point in front of camera
   j's measured surface (j sees the wall through it) is only *weak* support: either j's
   mask is wrong or the point is a calibration / registration spill. Points behind j's
   surface are occluded and say nothing. The count of supporting cameras per pixel is
   kept, not the number of points.
3. **Restore.** A pixel outside camera j's mask but within ``--band-px`` of it is restored
   when at least ``--min-views`` other cameras support it, or (``--keep-volume``) when its
   own edge-safe depth is subject volume (above the floor, inside the stage, not behind the
   local subject: refine_masks.depth_evidence) and at least one other camera supports it.
   Only regions connected to the mask are restored; ``--min-width-px`` > 0 also drops thin
   slivers. The restored region is upsampled, snapped to colour edges with the same guided
   filter as ``refine_masks.py``, feathered and merged with the original mask (the original
   mask is never reduced).
4. **Unknown band** (``--unknown``). Pixels near the mask where the evidence disagrees but
   is not strong enough to restore (weak support or a single supporting camera), and
   enclosed no-depth blobs next to the mask (black chair shell), are written as *unknown*
   in an alpha channel, see below.

Mask file contract (backward compatible). Without ``--unknown`` the output is a plain
8-bit greyscale PNG, as before. With ``--unknown`` frames that have unknown pixels are
written as 8-bit grey+alpha PNGs (PIL mode ``LA``): grey = the mask exactly as before
(0 = background, 255 = subject, soft edge), alpha = confidence (255 = known, 0 = unknown,
intermediate = soft weight). Readers that load with ``cv2.IMREAD_GRAYSCALE`` or PIL
``.convert("L")`` see the plain mask. A plain greyscale PNG means "all known".

Inputs (capture data directory): ``images/``, the per-camera masks (``--masks-dir``,
default ``masks_refined``), ``depth_registered/``, ``reference_export.json`` and optionally
``mask_report/masks.json`` (floor plane + stage). Outputs: ``--output-dir`` (default
``masks_consistent``) and ``<report-dir>/consistent.json`` + ``consistent_sheet.jpg``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rm = _load("refine_masks")
dm = rm.dm
_disk = dm._disk


# ======================================================================================
# Parameters
# ======================================================================================
@dataclass
class ConsistentParams:
    src_erode_px: int = 2  # depth px: erosion of each source mask before back-projection
    edge_tol: float = 0.03  # m: max depth range in a 3x3 window for an edge-safe depth pixel
    depth_tol: float = 0.04  # m: |projected z - measured z| that counts as agreement
    floor_margin: float = 0.05  # m above the floor plane: lower depth is not subject volume
    stage_margin: float = 0.10  # m added to the stage-cylinder radius for source points
    behind_margin: float = 0.25  # m behind the local subject surface = background (as refine_masks.py)
    local_radius_px: int = 40  # depth px: neighbourhood of the local subject depth
    min_views: int = 2  # other cameras that must support a pixel before it is restored
    band_px: int = 16  # depth px: restore / unknown only this close to the existing mask
    splat_px: int = 1  # depth px: closing radius of each camera's reprojected points
    min_width_px: int = 0  # depth px: opening radius of the restore map (0 = keep slivers)
    keep_volume: bool = True  # restore own subject-volume depth confirmed by >= 1 other camera
    unknown: bool = False  # write grey+alpha PNGs with an unknown band
    unknown_weak: bool = True  # weak support (camera sees background through the points) -> unknown
    unknown_nodepth_holes: bool = True  # enclosed no-depth blobs next to the mask -> unknown
    nodepth_max_frac: float = 0.02  # of the depth image: larger no-depth blobs are left alone
    unknown_dilate_px: int = 1  # depth px
    guided_radius: int = 4  # full-res px (as refine_masks.py)
    guided_eps: float = 1e-3
    feather_px: float = 1.0
    max_hole_frac: float = 0.003


# ======================================================================================
# Geometry
# ======================================================================================
def world_points(geo, z: np.ndarray, valid: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """World coordinates [N,3] of the selected depth pixels (depth resolution)."""
    r = rm._rays(geo, *z.shape)
    zz = z[sel]
    Xc = np.concatenate([r[sel] * zz[:, None], zz[:, None]], axis=1)
    return ((Xc - geo.t.astype(np.float32)) @ geo.R.astype(np.float32)).astype(np.float32)


def edge_safe(z: np.ndarray, valid: np.ndarray, tol: float) -> np.ndarray:
    """Valid depth pixels whose 3x3 neighbourhood is all valid and spans less than ``tol``."""
    k = np.ones((3, 3), np.uint8)
    hi = cv2.dilate(np.where(valid, z, 1e3).astype(np.float32), k)
    lo = cv2.erode(np.where(valid, z, -1e3).astype(np.float32), k)
    return valid & ((hi - lo) < tol)


def source_selection(geo, mask_d: np.ndarray, z: np.ndarray, valid: np.ndarray, p: ConsistentParams) -> np.ndarray:
    """Edge-safe, in-volume depth pixels inside the eroded mask (bool, depth resolution)."""
    sel = cv2.erode((mask_d > 0.5).astype(np.uint8), _disk(p.src_erode_px)) > 0
    sel &= edge_safe(z, valid, p.edge_tol)
    if not sel.any():
        return sel
    X = world_points(geo, z, valid, sel)
    keep = np.ones(len(X), bool)
    if geo.up is not None:
        keep &= X @ geo.up.astype(np.float32) + np.float32(geo.d - geo.bias) > p.floor_margin
        if geo.stage is not None and geo.e1 is not None:
            a = X @ geo.e1.astype(np.float32) - geo.stage[0]
            b = X @ geo.e2.astype(np.float32) - geo.stage[1]
            keep &= np.hypot(a, b) <= geo.stage[2] + p.stage_margin
    out = np.zeros_like(sel)
    out[sel] = keep
    return out


def project_depth_grid(geo, X: np.ndarray, h: int, w: int):
    """Project world points into a camera's depth grid (h x w covering the full image).
    Returns (u, v, z) for the points in front of the camera and inside the grid."""
    Xc = X @ geo.R.T.astype(np.float32) + geo.t.astype(np.float32)
    z = Xc[:, 2]
    ok = z > 0.05
    Xc, z = Xc[ok], z[ok]
    if np.any(geo.dist != 0):
        uv = cv2.projectPoints(Xc.astype(np.float64), np.zeros(3), np.zeros(3), geo.K, geo.dist)[0][:, 0]
    else:
        uv = np.c_[geo.K[0, 0] * Xc[:, 0] / z + geo.K[0, 2], geo.K[1, 1] * Xc[:, 1] / z + geo.K[1, 2]]
    u = np.round((uv[:, 0] + 0.5) * w / geo.width - 0.5).astype(np.int32)
    v = np.round((uv[:, 1] + 0.5) * h / geo.height - 0.5).astype(np.int32)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[inb], v[inb], z[inb]


def support_maps(views: list[dict], p: ConsistentParams):
    """Per camera: (strong, weak) uint8 maps at depth resolution = number of other cameras
    whose edge-safe in-mask depth points land there and agree with (or are unchecked by)
    this camera's depth (strong), or lie in front of its measured surface (weak only)."""
    pts = []
    for v in views:
        sel = source_selection(v["geo"], v["mask_d"], v["z"], v["valid"], p)
        pts.append(world_points(v["geo"], v["z"], v["valid"], sel) if sel.any() else np.zeros((0, 3), np.float32))
    k = _disk(p.splat_px)
    out = []
    for j, vj in enumerate(views):
        h, w = vj["z"].shape
        strong = np.zeros((h, w), np.uint8)
        weak = np.zeros((h, w), np.uint8)
        for i in range(len(views)):
            if i == j or len(pts[i]) == 0:
                continue
            u, v, zp = project_depth_grid(vj["geo"], pts[i], h, w)
            dj = vj["z"][v, u]
            has = vj["valid"][v, u]
            ok = ~has | (np.abs(zp - dj) < p.depth_tol)
            front = has & (zp < dj - p.depth_tol)
            s = np.zeros((h, w), np.uint8)
            s[v[ok], u[ok]] = 1
            f = np.zeros((h, w), np.uint8)
            f[v[front], u[front]] = 1
            if p.splat_px > 0:
                s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, k)
                f = cv2.morphologyEx(f, cv2.MORPH_CLOSE, k)
            strong += s
            weak += (f > 0) & (s == 0)
        out.append((strong, weak))
    return out


# ======================================================================================
# Per-frame decision (depth resolution)
# ======================================================================================
def prepare_view(geo, mask: np.ndarray, depth_m: np.ndarray) -> dict:
    """Depth maps and the mask at depth resolution for one camera."""
    z, valid, hgt, outside = rm.depth_maps(depth_m, geo)
    mask_d = cv2.resize(mask.astype(np.float32) / 255.0, (z.shape[1], z.shape[0]), interpolation=cv2.INTER_AREA)
    return dict(geo=geo, mask_d=mask_d, z=z, valid=valid, hgt=hgt, outside=outside)


def _touching(cand: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Connected components (8-conn.) of ``cand`` that intersect ``seed``."""
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(cand, dtype=bool)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[seed & cand])] = True
    keep[0] = False
    return keep[lab]


def nodepth_blobs(m: np.ndarray, valid: np.ndarray, bg: np.ndarray, hgt: np.ndarray | None,
                  p: ConsistentParams) -> np.ndarray:
    """Enclosed no-depth regions next to the mask: connected regions of invalid depth outside
    the mask that do not touch the image border, are smaller than ``nodepth_max_frac`` of the
    image, border the mask on at least 15% of their rim and have (almost) no background depth
    above floor level on their rim (floor seen at a grazing angle behind an object counts as
    "behind the subject" but is no sign of a see-through gap). Typically black fabric or
    plastic in front of the floor (a chair shell), which ToF cannot measure; a gap that looks
    through to a measured wall is not one."""
    if hgt is not None:
        bg = bg & (hgt > 2 * p.floor_margin)
    h, w = m.shape
    cand = (~valid & ~m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    out = np.zeros_like(m)
    near = np.unique(lab[(cv2.dilate(m.astype(np.uint8), _disk(2)) > 0) & (cand > 0)])
    for i in near:
        if i == 0:
            continue
        x, y, bw, bh, area = st[i]
        if x == 0 or y == 0 or x + bw == w or y + bh == h or area < 20 or area > p.nodepth_max_frac * h * w:
            continue
        y0, y1, x0, x1 = max(0, y - 3), min(h, y + bh + 3), max(0, x - 3), min(w, x + bw + 3)
        r = lab[y0:y1, x0:x1] == i
        rim = (cv2.dilate(r.astype(np.uint8), _disk(2)) > 0) & ~r
        nrim = max(1, int(rim.sum()))
        if (rim & m[y0:y1, x0:x1]).sum() < 0.15 * nrim:
            continue
        if (rim & bg[y0:y1, x0:x1]).sum() > 0.1 * nrim:
            continue
        out[y0:y1, x0:x1] |= r
    return out


def decide_view(v: dict, strong: np.ndarray, weak: np.ndarray, p: ConsistentParams):
    """(restore, unknown, info) at depth resolution for one camera."""
    m = v["mask_d"] > 0.5
    if not m.any():
        z = np.zeros_like(m)
        return z, z.copy(), {"restore_px": 0}
    band = (cv2.dilate(m.astype(np.uint8), _disk(p.band_px)) > 0) & ~m
    rp = rm.RefineParams(floor_margin=p.floor_margin, behind_margin=p.behind_margin, local_radius_px=p.local_radius_px)
    core = cv2.erode(m.astype(np.uint8), _disk(4)) > 0
    bg, floor, vol = rm.depth_evidence(v["z"], v["valid"], v["hgt"], v["outside"], band | m, core, rp)
    vol &= edge_safe(v["z"], v["valid"], p.edge_tol)
    restore = band & (strong >= p.min_views)
    if p.keep_volume:
        restore |= band & vol & (strong >= 1)
    restore &= ~bg
    if p.min_width_px > 0:
        restore = cv2.morphologyEx(restore.astype(np.uint8), cv2.MORPH_OPEN, _disk(p.min_width_px)) > 0
    restore = _touching(restore, cv2.dilate(m.astype(np.uint8), _disk(1)) > 0)
    unknown = np.zeros_like(m)
    info = {"restore_px": int(restore.sum())}
    if p.unknown:
        unknown = band & (strong >= 1)
        if p.unknown_weak:
            unknown |= band & (weak >= 1)
        if p.unknown_dilate_px > 0:
            unknown = (cv2.dilate(unknown.astype(np.uint8), _disk(p.unknown_dilate_px)) > 0) & band
        if p.unknown_nodepth_holes:
            holes = nodepth_blobs(m, v["valid"], bg, v["hgt"], p)
            info["nodepth_px"] = int(holes.sum())
            unknown |= holes
        unknown &= ~restore & ~m
    return restore, unknown, info


# ======================================================================================
# Full-resolution finishing
# ======================================================================================
def finish_view(mask: np.ndarray, img: np.ndarray, restore_d: np.ndarray, unknown_d: np.ndarray,
                p: ConsistentParams):
    """Merge the restored region into the full-resolution mask; returns (grey uint8,
    confidence uint8 or None). The input mask is never reduced; it changes only within a
    few pixels of the restored region."""
    H, W = mask.shape[:2]
    grey = mask.copy()
    if restore_d.any():
        prob = cv2.resize(restore_d.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        if p.guided_radius > 0:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            prob = rm.guided_filter(g, prob, p.guided_radius, p.guided_eps)
        allowed = cv2.resize(cv2.dilate(restore_d.astype(np.uint8), _disk(1)), (W, H), interpolation=cv2.INTER_NEAREST)
        old = mask >= 128
        hard = old | ((prob > 0.5) & (allowed > 0))
        hard = dm.fill_holes(hard.astype(np.uint8), p.max_hole_frac * H * W) > 0
        add = hard & ~old
        if add.any():
            soft = hard.astype(np.float32)
            if p.feather_px > 0:
                soft = np.maximum(cv2.GaussianBlur(soft, (0, 0), p.feather_px), hard * 0.5)
            near = cv2.dilate(add.astype(np.uint8), _disk(int(np.ceil(3 * p.feather_px)) + 1)) > 0
            new = np.clip(soft * 255.0 + 0.5, 0, 255).astype(np.uint8)
            grey = np.where(near, np.maximum(mask, new), mask)
    conf = None
    if p.unknown and unknown_d.any():
        u = cv2.resize(unknown_d.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        u &= grey < 128
        if u.any():
            conf = np.where(u, 0, 255).astype(np.uint8)
    return grey, conf


def consistent_frame(views: list[dict], masks: list[np.ndarray], imgs: list[np.ndarray], p: ConsistentParams):
    """All cameras of one frame -> list of (grey, conf or None, info)."""
    sup = support_maps(views, p)
    out = []
    for v, (strong, weak), m, img in zip(views, sup, masks, imgs):
        restore, unknown, info = decide_view(v, strong, weak, p)
        grey, conf = finish_view(m, img, restore, unknown, p)
        info.update(area_in=float((m >= 128).mean()), area_out=float((grey >= 128).mean()),
                    unknown_frac=float((conf == 0).mean()) if conf is not None else 0.0)
        out.append((grey, conf, info))
    return out


# ======================================================================================
# Mask files (plain grey or grey + confidence alpha)
# ======================================================================================
def write_mask(path: Path, grey: np.ndarray, conf: np.ndarray | None) -> None:
    """Plain 8-bit greyscale PNG, or grey+alpha (PIL "LA") when ``conf`` is given."""
    if conf is None:
        if not cv2.imwrite(str(path), grey):
            raise OSError(f"could not write {path}")
        return
    from PIL import Image

    Image.fromarray(np.dstack([grey, conf]), mode="LA").save(str(path))


def read_mask(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(grey, confidence or None) of a plain or grey+alpha mask PNG."""
    from PIL import Image

    if not Path(path).exists():
        return None, None
    with Image.open(path) as im:
        if im.mode == "LA":
            a = np.asarray(im)
            return a[..., 0].copy(), a[..., 1].copy()
        return np.asarray(im.convert("L")).copy(), None


# ======================================================================================
# Workers
# ======================================================================================
_W: dict = {}


def _init(cfg: dict):
    _W.clear()
    _W.update(cfg)
    _W["params"] = ConsistentParams(**cfg["params"])


def _do_frame(f: int):
    p: ConsistentParams = _W["params"]
    views, masks, imgs, cams = [], [], [], []
    for c in _W["cams"]:
        img = cv2.imread(str(_W["images"] / c / f"{f:06d}.{_W['ext']}"))
        m = cv2.imread(str(_W["masks"] / c / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        d = rm.read_depth(_W["depth"] / c / f"{f:06d}", _W["depth_scale"])
        if d is None:  # no depth: this camera neither gives nor receives depth-checked support
            d = np.zeros((max(1, m.shape[0] // 2), max(1, m.shape[1] // 2)), np.float32)
        views.append(prepare_view(_W["geos"][c], m, d))
        masks.append(m)
        imgs.append(img)
        cams.append(c)
    rows = []
    for c, (grey, conf, info) in zip(cams, consistent_frame(views, masks, imgs, p)):
        write_mask(_W["out"] / c / f"{f:06d}.png", grey, conf)
        rows.append((c, dict(frame=f, **info)))
    return f, rows


# ======================================================================================
# QA
# ======================================================================================
def qa_sheet(cams, images_root: Path, in_root: Path, out_root: Path, frames, path: Path, ext: str = "jpg",
             tile_h: int = 640):
    """Background dimmed by the output mask; added pixels green, unknown magenta, input
    mask outline white."""
    rows = []
    for f in frames:
        tiles = []
        for c in cams:
            img = cv2.imread(str(images_root / c / f"{f:06d}.{ext}"))
            m0 = cv2.imread(str(in_root / c / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
            m1, conf = read_mask(out_root / c / f"{f:06d}.png")
            if img is None or m0 is None or m1 is None:
                continue
            a = (m1.astype(np.float32) / 255.0)[..., None]
            comp = img * a + img * 0.3 * (1 - a)
            add = (m1 >= 128) & (m0 < 128)
            comp[add] = comp[add] * 0.5 + np.array([0, 200, 0]) * 0.5
            if conf is not None:
                u = conf < 128
                comp[u] = comp[u] * 0.4 + np.array([200, 0, 200]) * 0.6
            comp = comp.clip(0, 255).astype(np.uint8)
            cnts, _ = cv2.findContours((m0 >= 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(comp, cnts, -1, (255, 255, 255), 1)
            ys, xs = np.nonzero(m1 >= 128)
            if len(ys):
                pad = 60
                comp = comp[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]
            cv2.putText(comp, f"{c} f{f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            s = tile_h / comp.shape[0]
            tiles.append(cv2.resize(comp, (max(1, int(comp.shape[1] * s)), tile_h), interpolation=cv2.INTER_AREA))
        if tiles:
            rows.append(np.hstack(tiles))
    if rows:
        w = max(r.shape[1] for r in rows)
        cv2.imwrite(str(path), np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT)
                                          for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 88])


# ======================================================================================
# CLI
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--images-dir", default="images")
    ap.add_argument("--image-ext", default="jpg")
    ap.add_argument("--masks-dir", default="masks_refined", help="per-camera masks (input)")
    ap.add_argument("--output-dir", default=None, help="default: <data-dir>/masks_consistent")
    ap.add_argument("--report-dir", default=None, help="default: <data-dir>/mask_report")
    ap.add_argument("--report-name", default="consistent", help="<report-dir>/<name>.json and <name>_sheet.jpg")
    ap.add_argument("--reference-json", default="reference_export.json")
    ap.add_argument("--depth-dir", default="depth_registered")
    ap.add_argument("--depth-scale", type=float, default=0.001, help="metres per PNG unit")
    ap.add_argument("--ground-json", default="mask_report/masks.json",
                    help="depth_masks report with the floor plane and stage (relative to --data-dir)")
    ap.add_argument("--cameras", default=None, help="comma-separated camera folders (default: all with masks)")
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None, help="inclusive")
    ap.add_argument("--frames", default=None, help="explicit list, e.g. 1-31,40,60,90 (overrides start/end)")
    ap.add_argument("--qa-frames", default=None, help="frames on the QA sheet (default: 4 spread over the run)")
    for k, v in asdict(ConsistentParams()).items():
        name = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(name, action=argparse.BooleanOptionalAction, default=v)
        else:
            ap.add_argument(name, type=type(v), default=v)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="frames processed in parallel (each holds all cameras of one frame)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data = Path(args.data_dir)
    images_root = data / args.images_dir
    in_root = data / args.masks_dir
    out_root = Path(args.output_dir) if args.output_dir else data / "masks_consistent"
    report_dir = Path(args.report_dir) if args.report_dir else data / "mask_report"
    if out_root.resolve() == in_root.resolve():
        raise SystemExit("--output-dir must differ from --masks-dir")
    depth_root = data / args.depth_dir
    if not depth_root.is_dir() or not (data / args.reference_json).exists():
        raise SystemExit("the consistency pass needs registered depth (--depth-dir) and camera poses (--reference-json)")
    report_dir.mkdir(parents=True, exist_ok=True)
    p = ConsistentParams(**{k: getattr(args, k) for k in asdict(ConsistentParams())})

    cams = sorted(d.name for d in in_root.iterdir() if d.is_dir())
    if args.cameras:
        cams = [c for c in args.cameras.split(",") if c in cams]
    if len(cams) < 2:
        raise SystemExit(f"need at least 2 camera folders in {in_root}")
    nums = None
    for c in cams:
        s = {int(q.stem) for q in (in_root / c).glob("*.png") if q.stem.isdigit()}
        nums = s if nums is None else nums & s
    if not nums:
        raise SystemExit("no frame has a mask in every camera")
    if args.frames:
        frames = [f for f in rm.parse_frames(args.frames) if f in nums]
    else:
        lo = min(nums) if args.frame_start is None else args.frame_start
        hi = max(nums) if args.frame_end is None else args.frame_end
        frames = [f for f in sorted(nums) if lo <= f <= hi]
    if not frames:
        raise SystemExit("no frames selected")
    qa = rm.parse_frames(args.qa_frames) if args.qa_frames else \
        sorted({frames[int(round(i))] for i in np.linspace(0, len(frames) - 1, 4)})
    qa = [f for f in qa if f in frames]

    gj = Path(args.ground_json)
    gj = gj if gj.is_absolute() else data / gj
    geos, ginfo = rm.build_geometries(data, args.reference_json, depth_root, args.depth_scale, gj, frames, cams)
    print(f"{len(cams)} cameras, {len(frames)} frames, floor: {ginfo['floor']}, stage: {ginfo['stage']}")

    for c in cams:
        (out_root / c).mkdir(parents=True, exist_ok=True)
    cfg = {"images": images_root, "masks": in_root, "depth": depth_root, "depth_scale": args.depth_scale,
           "ext": args.image_ext, "out": out_root, "geos": geos, "params": asdict(p), "cams": cams}
    t0 = time.time()
    per_cam: dict[str, list] = {c: [] for c in cams}
    with contextlib.ExitStack() as stack:
        if args.workers > 1:
            pool = stack.enter_context(ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                                           initargs=(cfg,)))
            results = pool.map(_do_frame, frames)
        else:  # in-process (also used by the tests)
            _init(cfg)
            results = map(_do_frame, frames)
        for i, (f, rows) in enumerate(results, 1):
            for c, r in rows:
                per_cam[c].append(r)
            if i % 20 == 0 or i == len(frames):
                print(f"  {i}/{len(frames)} frames ({time.time() - t0:.0f} s)", flush=True)
    wall = time.time() - t0

    qa_sheet(cams, images_root, in_root, out_root, qa, report_dir / f"{args.report_name}_sheet.jpg", args.image_ext)
    summary = {}
    for c in cams:
        rows = sorted(per_cam[c], key=lambda r: r["frame"])
        if not rows:
            continue
        a0 = np.array([r["area_in"] for r in rows])
        a1 = np.array([r["area_out"] for r in rows])
        uf = np.array([r["unknown_frac"] for r in rows])
        gain = (a1 - a0) / np.maximum(a0, 1e-9) * 100
        ti = rm.temporal_iou(out_root, c, [r["frame"] for r in rows])
        summary[c] = {"area_in_median": float(np.median(a0)), "area_out_median": float(np.median(a1)),
                      "area_gain_pct_median": float(np.median(gain)), "area_gain_pct_max": float(np.max(gain)),
                      "unknown_frac_of_image_mean": float(uf.mean()),
                      "unknown_frac_of_mask_mean": float(np.mean(uf / np.maximum(a1, 1e-9))),
                      "frames_with_unknown": int((uf > 0).sum()),
                      "temporal_iou_mean": float(np.mean(ti)) if ti else None}
        s = summary[c]
        print(f"  {c}: area {s['area_in_median'] * 100:.2f}% -> {s['area_out_median'] * 100:.2f}% of the image "
              f"(+{s['area_gain_pct_median']:.1f}% median, max +{s['area_gain_pct_max']:.1f}%); unknown "
              f"{s['unknown_frac_of_image_mean'] * 100:.2f}% of the image "
              f"({s['unknown_frac_of_mask_mean'] * 100:.1f}% of the mask area)")
    # straight_foregrounds.py erodes by dilate_px + 2 * feather_px from the report whose
    # output_dir matches its --masks-dir; the solid core here is that of the input masks.
    dilate_px = 1
    for rep in sorted(report_dir.glob("*.json")):
        try:
            j = json.loads(rep.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if Path(str(j.get("output_dir", ""))).name == in_root.name and "dilate_px" in j.get("params", {}):
            dilate_px = int(j["params"]["dilate_px"])
            break
    params = asdict(p)
    params["dilate_px"] = dilate_px
    report = {"params": params, "input_dir": str(in_root), "output_dir": str(out_root), "depth": str(depth_root),
              "ground": ginfo, "frames": frames if len(frames) < 64 else [frames[0], frames[-1], len(frames)],
              "qa_frames": qa, "wall_seconds": wall, "masks": sum(len(v) for v in per_cam.values()),
              "cameras": summary}
    (report_dir / f"{args.report_name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(frames)} frames in {wall:.0f} s ({wall / len(frames):.2f} s per frame wall). "
          f"Masks: {out_root}  QA: {report_dir / (args.report_name + '_sheet.jpg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
