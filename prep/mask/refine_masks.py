#!/usr/bin/env python3
"""Refine coarse foreground masks with colour (and, optionally, registered depth).

Coarse masks from depth (``depth_masks.py``) or any other source are usually loose: they
include floor around the feet and pieces of wall or light stands next to the subject. This
script shrinks every coarse mask to the colour edge of the subject. It never adds anything
more than ``--max-grow-px`` outside the coarse mask.

Inputs (capture data directory)::

    images/<cam>/<frame:06d>.jpg         colour images
    masks/<cam>/<frame:06d>.png          coarse 8-bit masks (255 = subject), any source
    optional, for the depth cues:
    reference_export.json                camera intrinsics + world->camera poses (metres)
    depth_registered/<cam>/<frame:06d>.png|.npy
                                         z-depth registered to the colour camera (same
                                         contract as depth_masks.py --depth-source registered)
    mask_report/masks.json               floor plane + per-camera floor offsets + stage
                                         cylinder written by depth_masks.py (otherwise the
                                         floor is fitted from the registered depth)

Per camera and frame, at ``--work-scale`` of the image resolution:

1. Depth cues (when depth is available). A pixel with valid depth is *background evidence*
   when it lies on the floor (height < ``--floor-margin``), outside the stage cylinder
   (when masks.json has one), or more than ``--behind-margin`` behind the farthest subject
   surface within ``--local-radius-px`` (subject depth taken from the eroded core of the
   coarse mask). It is *foreground evidence* when it is inside the coarse mask, above the
   floor and not behind the subject.
2. Trimap for GrabCut: sure background = outside the coarse mask (dilated by
   ``--max-grow-px``) and the eroded depth-background evidence; sure foreground = the coarse
   mask eroded by ``--core-erode-px`` minus depth-background, plus the eroded
   depth-foreground evidence; everything else in the coarse mask is "probably
   foreground", depth-background edges are "probably background".
3. GrabCut (colour GMMs + contrast-sensitive smoothness) labels the uncertain pixels.
4. Temporal vote: the binary results of ``--temporal`` frames on each side are averaged.
5. At full resolution: guided-filter upsampling on the image (edges snap to colour edges),
   threshold, cut to the dilated coarse mask, keep the components that contain sure
   foreground, fill small enclosed holes, ``--dilate-px`` safety margin and a Gaussian
   feather of ``--feather-px``.

Outputs: ``masks_refined/<cam>/<frame:06d>.png`` (same size and polarity as the coarse
masks), ``<report-dir>/refine.json`` and ``<report-dir>/refine_sheet.jpg``.
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


def _load_depth_masks():
    name = "depth_masks"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "depth_masks.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dm = _load_depth_masks()
_disk = dm._disk


# ======================================================================================
# Parameters
# ======================================================================================
@dataclass
class RefineParams:
    work_scale: float = 0.5  # segmentation resolution relative to the image
    core_erode_px: int = 12  # work px: coarse mask erosion that gives the sure-foreground core
    max_grow_px: int = 2  # work px: the refined mask may extend this far outside the coarse mask
    roi_pad_px: int = 40  # work px: background context around the coarse mask for GrabCut
    grabcut_iters: int = 4
    floor_margin: float = 0.03  # m above the floor plane that still counts as floor
    lower_body_m: float = 0.5  # m: sure-foreground below this height is the "lower body" colour model
    floor_sure_bg_prob: float = 0.1  # floor-level pixels with P(fg | colour) below this are sure background
    person_erode_px: int = 2  # work px erosion of the SAM 2 person mask before it becomes sure foreground
    person_grow_px: int = 20  # work px: the person mask may extend the coarse mask this far
    unclaimed_gap_px: int = 4  # work px; 0 = off. See segment_frame.
    behind_margin: float = 0.25  # m behind the local subject surface that counts as background
    stage_margin: float = 0.10  # m added to the stage-cylinder radius before it rejects pixels
    local_radius_px: int = 40  # work px: neighbourhood for the local subject depth
    fg_evidence_erode_px: int = 2  # work px erosion of depth evidence before it becomes "sure"
    bg_evidence_erode_px: int = 2
    temporal: int = 1  # frames on each side in the temporal vote (0 = off)
    guided_radius: int = 4  # full-res px
    guided_eps: float = 1e-3
    dilate_px: int = 1  # full-res px safety margin
    feather_px: float = 1.0  # Gaussian sigma (full-res px)
    max_hole_frac: float = 0.003
    use_depth: bool = True


GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD = cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD


# ======================================================================================
# Depth cues
# ======================================================================================
@dataclass
class DepthGeometry:
    """What is needed to turn one camera's registered depth map into depth cues."""

    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    R: np.ndarray  # world -> camera
    t: np.ndarray
    up: np.ndarray | None = None  # floor plane: height = up . X + d - bias
    d: float = 0.0
    bias: float = 0.0
    stage: tuple[float, float, float] | None = None  # (c1, c2, radius) in (e1, e2) ground coords
    e1: np.ndarray | None = None
    e2: np.ndarray | None = None


_RAYS: dict = {}


def _rays(geo: DepthGeometry, h: int, w: int) -> np.ndarray:
    key = (id(geo), h, w)
    if key not in _RAYS:
        if abs(w / h - geo.width / geo.height) > 0.01:
            raise ValueError(f"depth {w}x{h} does not match the image aspect {geo.width}x{geo.height}")
        sx, sy = geo.width / w, geo.height / h
        uu, vv = np.meshgrid((np.arange(w) + 0.5) * sx - 0.5, (np.arange(h) + 0.5) * sy - 0.5)
        pix = np.stack([uu.ravel(), vv.ravel()], 1)[:, None].astype(np.float64)
        _RAYS[key] = cv2.undistortPoints(pix, geo.K, geo.dist)[:, 0].reshape(h, w, 2).astype(np.float32)
    return _RAYS[key]


def depth_maps(depth_m: np.ndarray, geo: DepthGeometry | None):
    """Per-pixel (z, valid, height above floor or None, outside-stage or None) at depth resolution."""
    z = depth_m.astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    z = np.where(valid, z, 0).astype(np.float32)
    if geo is None or geo.up is None:
        return z, valid, None, None
    r = _rays(geo, *z.shape)
    Xc = np.concatenate([r * z[..., None], z[..., None]], axis=2).reshape(-1, 3)
    Xw = (Xc - geo.t.astype(np.float32)) @ geo.R.astype(np.float32)
    hgt = (Xw @ geo.up.astype(np.float32) + np.float32(geo.d - geo.bias)).reshape(z.shape)
    outside = None
    if geo.stage is not None and geo.e1 is not None:
        a = Xw @ geo.e1.astype(np.float32) - geo.stage[0]
        b = Xw @ geo.e2.astype(np.float32) - geo.stage[1]
        outside = (np.hypot(a, b) > geo.stage[2]).reshape(z.shape)
    return z, valid, hgt, outside


def depth_evidence(z, valid, hgt, outside, coarse: np.ndarray, core: np.ndarray, p: RefineParams):
    """Evidence maps (bool, work resolution) from registered depth: (bg, floor, fg).

    ``bg``: off the stage or clearly behind the subject (reliable). ``floor``: at floor
    height. ToF depth cannot tell a shoe from the floor it stands on (shoe toes measure
    0-3 cm, the floor scatters by +-3 cm), so floor-level pixels are only *candidates*;
    colour decides them."""
    bg = np.zeros_like(valid)
    floor = np.zeros_like(valid)
    if hgt is not None:
        floor = valid & (hgt < p.floor_margin)
    if outside is not None:
        bg |= valid & outside
    # Subject depth: valid depth in the core that is not already background, robust to thick
    # wall patches that survived the erosion (> 1 m behind the median subject depth).
    cz = valid & core & ~bg & ~floor
    if cz.sum() >= 50:
        med = float(np.median(z[cz]))
        cz &= z < med + 1.0
    if cz.sum() >= 50:
        far_global = float(np.percentile(z[cz], 98))
        ref = cv2.dilate(np.where(cz, z, 0).astype(np.float32), _disk(p.local_radius_px))
        ref = np.where(ref > 0, ref, far_global)
        bg |= valid & (z > ref + p.behind_margin)
    floor &= ~bg
    fg = valid & coarse & ~bg & ~floor
    if hgt is not None:
        fg &= hgt > max(2 * p.floor_margin, 0.05)
    return bg, floor, fg


def colour_fg_probability(im: np.ndarray, fg_sel: np.ndarray, bg_sel: np.ndarray, bins: int = 16) -> np.ndarray | None:
    """P(fg | colour) per pixel from smoothed 3D colour histograms of two sample sets
    (equal priors). None when either set is too small."""
    if fg_sel.sum() < 100 or bg_sel.sum() < 100:
        return None
    q = (im.astype(np.int32) * bins) // 256
    idx = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]

    def hist(sel):
        h = np.bincount(idx[sel], minlength=bins ** 3).astype(np.float32).reshape(bins, bins, bins)
        k = np.array([0.25, 0.5, 0.25], np.float32)
        for ax in range(3):
            h = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, h)
        return h / h.sum() + 1e-6

    hf, hb = hist(fg_sel), hist(bg_sel)
    pf = hf.ravel()[idx]
    pb = hb.ravel()[idx]
    return pf / (pf + pb)


# ======================================================================================
# Per-frame segmentation
# ======================================================================================
def _resize(a: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    if (a.shape[1], a.shape[0]) == size:
        return a
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(a, size, interpolation=interp)


def segment_frame(img: np.ndarray, coarse: np.ndarray, depth_m: np.ndarray | None,
                  geo: DepthGeometry | None, p: RefineParams, person: np.ndarray | None = None):
    """One frame at work resolution. ``person``: optional uint8 mask (image size) from a
    segmentation network; its eroded interior becomes sure foreground.

    Returns (fg, info): fg is a uint8 0/1 map at work resolution, info holds the trimap and
    statistics."""
    H, W = coarse.shape[:2]
    ws = (max(1, int(round(W * p.work_scale))), max(1, int(round(H * p.work_scale))))
    im = _resize(img, ws)
    c = (_resize(coarse, ws) > 127).astype(np.uint8)
    info: dict = {"depth": False}
    if c.sum() == 0:
        return c, info
    core = cv2.erode(c, _disk(p.core_erode_px)) > 0
    allowed = cv2.dilate(c, _disk(p.max_grow_px)) > 0

    dbg = np.zeros(c.shape, bool)
    dfloor = np.zeros(c.shape, bool)
    dfg = np.zeros(c.shape, bool)
    hgt = None
    if depth_m is not None and p.use_depth:
        z, valid, hgt, outside = depth_maps(depth_m, geo)
        maps = [z, valid.astype(np.uint8)] + [m for m in (hgt, outside) if m is not None]
        maps = [_resize(m if m.dtype != bool else m.astype(np.uint8), ws, nearest=True) for m in maps]
        z, valid = maps[0], maps[1] > 0
        k = 2
        if hgt is not None:
            hgt = maps[k]
            k += 1
        if outside is not None:
            outside = maps[k] > 0
        dbg, dfloor, dfg = depth_evidence(z, valid, hgt, outside, c > 0, core, p)
        info["depth"] = True

    gc = np.full(c.shape, GC_BGD, np.uint8)
    gc[allowed] = GC_PR_FGD
    gc[allowed & dbg] = GC_PR_BGD
    sure_bg = cv2.erode(dbg.astype(np.uint8), _disk(p.bg_evidence_erode_px)) > 0
    gc[sure_bg] = GC_BGD
    not_core = cv2.dilate((dbg | dfloor).astype(np.uint8), _disk(2)) > 0
    sure_fg = (core & ~not_core) | (cv2.erode(dfg.astype(np.uint8), _disk(p.fg_evidence_erode_px)) > 0)

    pw = None
    if person is not None and person.any():
        pw = _resize((person > 127).astype(np.uint8), ws, nearest=True) > 0
        # The person mask may extend the coarse mask by up to person_grow_px (depth masks often
        # clip shoe toes at the floor margin), never onto depth background.
        grow = pw & (cv2.dilate(c, _disk(p.person_grow_px)) > 0) & ~dbg & ~allowed
        allowed = allowed | grow
        gc[grow] = GC_PR_FGD
        pw &= allowed
        info["person_frac_of_coarse"] = float(pw.sum() / max(1, c.sum()))
        info["grown_frac_of_coarse"] = float(grow.sum() / max(1, c.sum()))
        # Pixels that neither the person segmentation nor depth claim start as "probably
        # background" (loose patches without valid depth, e.g. dark walls).
        near_person = cv2.dilate(pw.astype(np.uint8), _disk(p.person_erode_px + 1)) > 0
        gc[allowed & ~near_person & ~dfg & ~core & (gc == GC_PR_FGD)] = GC_PR_BGD
        if p.unclaimed_gap_px > 0 and info["depth"]:
            # Claimed by nothing (outside the person mask by more than the gap, no subject depth
            # nearby, outside the core): sure background. Dark walls / stands next to the hair
            # look like hair, so colour alone does not remove them.
            far_person = ~(cv2.dilate(pw.astype(np.uint8), _disk(p.unclaimed_gap_px)) > 0)
            near_dfg = cv2.dilate(dfg.astype(np.uint8), _disk(p.fg_evidence_erode_px + 1)) > 0
            gc[allowed & far_person & ~near_dfg & ~core] = GC_BGD
        sure_fg |= (cv2.erode(pw.astype(np.uint8), _disk(p.person_erode_px)) > 0) & ~dbg
    sure_fg &= allowed
    gc[sure_fg] = GC_FGD

    # Floor-level candidates inside the coarse mask: initial label from a local colour model
    # (floor just outside the mask vs. the subject's lower body), then GrabCut decides.
    cand = allowed & dfloor & ~sure_fg & ~dbg
    if pw is not None:
        cand &= ~pw
    if cand.any():
        ring = (cv2.dilate(c, _disk(p.roi_pad_px)) > 0) & ~allowed & dfloor
        low = sure_fg & (hgt < p.lower_body_m) if hgt is not None else sure_fg
        if low.sum() < 200:
            low = sure_fg
        pf = colour_fg_probability(im, low, ring)
        if pf is not None:
            gc[cand & (pf < 0.5)] = GC_PR_BGD
            gc[cand & (pf < p.floor_sure_bg_prob)] = GC_BGD
    info["sure_fg_frac_of_coarse"] = float(sure_fg.sum() / max(1, c.sum()))
    info["depth_bg_frac_of_coarse"] = float((dbg & (c > 0)).sum() / max(1, c.sum()))
    info["floor_frac_of_coarse"] = float((dfloor & (c > 0)).sum() / max(1, c.sum()))

    # ROI around the coarse mask with background context.
    ys, xs = np.nonzero(allowed)
    pad = p.roi_pad_px
    y0, y1 = max(0, ys.min() - pad), min(c.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(c.shape[1], xs.max() + pad + 1)
    sub = gc[y0:y1, x0:x1].copy()
    has_bg = np.any((sub == GC_BGD) | (sub == GC_PR_BGD))
    has_fg = np.any((sub == GC_FGD) | (sub == GC_PR_FGD))
    if has_bg and has_fg and p.grabcut_iters > 0:
        bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(np.ascontiguousarray(im[y0:y1, x0:x1]), sub, None, bgm, fgm, p.grabcut_iters,
                        cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            pass  # degenerate colour statistics: keep the trimap labels
    gc[y0:y1, x0:x1] = sub
    fg = ((gc == GC_FGD) | (gc == GC_PR_FGD)).astype(np.uint8)
    fg &= allowed.astype(np.uint8)
    info["trimap"] = gc
    info["allowed"] = allowed
    info["sure_fg"] = sure_fg
    return fg, info


# ======================================================================================
# Full-resolution finishing
# ======================================================================================
def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Grey-guide guided filter (He et al.), box windows of size 2r+1."""
    I = guide.astype(np.float32)
    k = (2 * radius + 1, 2 * radius + 1)
    mI = cv2.boxFilter(I, -1, k)
    mp = cv2.boxFilter(src, -1, k)
    cov = cv2.boxFilter(I * src, -1, k) - mI * mp
    var = cv2.boxFilter(I * I, -1, k) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def finish_mask(prob_work: np.ndarray, img: np.ndarray, coarse: np.ndarray, sure_fg_work: np.ndarray | None,
                p: RefineParams, allowed_work: np.ndarray | None = None) -> np.ndarray:
    """Work-resolution foreground probability -> full-resolution uint8 alpha mask."""
    H, W = coarse.shape[:2]
    prob = cv2.resize(prob_work.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    if p.guided_radius > 0:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        prob = guided_filter(grey, prob, p.guided_radius, p.guided_eps)
    scale = 1.0 / p.work_scale
    allowed = cv2.dilate((coarse > 127).astype(np.uint8), _disk(int(round(p.max_grow_px * scale))))
    if allowed_work is not None:  # includes the person-mask growth
        allowed |= cv2.resize(allowed_work.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    m = ((prob > 0.5) & (allowed > 0)).astype(np.uint8)
    # Keep components that contain sure foreground (or, without it, the large ones).
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        keep = np.zeros(n, bool)
        if sure_fg_work is not None and sure_fg_work.any():
            sf = cv2.resize(sure_fg_work.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            keep[np.unique(lab[sf & (m > 0)])] = True
        else:
            keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 0.002 * H * W
        keep[0] = False
        m = keep[lab].astype(np.uint8)
    m = dm.fill_holes(m, p.max_hole_frac * H * W)
    if p.dilate_px > 0:
        m = cv2.dilate(m, _disk(p.dilate_px)) & allowed
    alpha = m.astype(np.float32)
    if p.feather_px > 0:
        alpha = np.maximum(cv2.GaussianBlur(alpha, (0, 0), p.feather_px), m * 0.5)
        alpha *= cv2.GaussianBlur(allowed.astype(np.float32), (0, 0), p.feather_px) > 0.01
    return np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)


def temporal_vote(fgs: list[np.ndarray]) -> np.ndarray:
    """Mean of the binary maps of an odd window of frames; > 0.5 is a majority vote, so a
    label that flips in one frame only (in either direction) is overruled."""
    return sum(f.astype(np.float32) for f in fgs) / float(len(fgs))


# ======================================================================================
# Data access
# ======================================================================================
def read_depth(path_base: Path, scale: float) -> np.ndarray | None:
    if path_base.with_suffix(".npy").exists():
        return np.load(path_base.with_suffix(".npy")).astype(np.float32)
    d = cv2.imread(str(path_base.with_suffix(".png")), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32) * scale


def build_geometries(data: Path, ref_json: str, depth_root: Path | None, depth_scale: float,
                     ground_json: Path | None, frames: list[int], cam_folders: list[str]):
    """Per-camera DepthGeometry (None entries when no poses). Floor plane from masks.json
    when present, else fitted from the registered depth."""
    ref = data / ref_json
    if depth_root is None or not ref.exists():
        return {f: None for f in cam_folders}, None
    cams = dm.load_cameras(data, ref_json)
    ground = None
    stage = None
    bias = [0.0] * len(cams)
    source_note = "none"
    if ground_json is not None and ground_json.exists():
        mj = json.loads(ground_json.read_text(encoding="utf-8"))
        g = mj.get("ground") or {}
        if g.get("has_floor"):
            ground = dm.Ground(np.asarray(g["up"]), float(g["d"]), np.asarray(g["e1"]), np.asarray(g["e2"]), True)
            prm = mj.get("params") or {}
            fb = prm.get("floor_bias") or []
            if len(fb) == len(cams):
                bias = [float(b) for b in fb]
            if prm.get("center") is not None and prm.get("radius") is not None:
                stage = (float(prm["center"][0]), float(prm["center"][1]), float(prm["radius"]))
            source_note = str(ground_json)
    if ground is None:
        try:
            src = dm.RegisteredDepthSource(data, cams, depth_root.name if depth_root.parent == data else str(depth_root),
                                           depth_scale)
            sample = frames[:: max(1, len(frames) // 3)][:3]
            ground = dm.estimate_ground(cams, src, sample)
            source_note = "fitted from registered depth"
        except (SystemExit, FileNotFoundError, ValueError) as e:
            print(f"  floor plane unavailable ({e}); depth cue limited to 'behind the subject'")
    out = {}
    for ci, c in enumerate(cams):
        geo = DepthGeometry(c.K, c.dist, c.width, c.height, c.R, c.t)
        if ground is not None:
            geo.up, geo.d, geo.bias = ground.up, ground.d, bias[ci]
            geo.e1, geo.e2 = ground.e1, ground.e2
            geo.stage = stage
        out[c.folder] = geo
    info = {"floor": source_note, "stage": list(stage) if stage else None,
            "up": ground.up.tolist() if ground is not None else None}
    return out, info


# ======================================================================================
# SAM 2 person masks (optional)
# ======================================================================================
SAM_CONFIGS = {"large": "configs/sam2.1/sam2.1_hiera_l.yaml", "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
               "small": "configs/sam2.1/sam2.1_hiera_s.yaml", "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml"}
SAM_FILES = {"large": "sam2.1_hiera_large.pt", "base_plus": "sam2.1_hiera_base_plus.pt",
             "small": "sam2.1_hiera_small.pt", "tiny": "sam2.1_hiera_tiny.pt"}


def find_sam_checkpoint(model: str, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit) if Path(explicit).exists() else None
    name = SAM_FILES[model]
    cands = []
    if os.environ.get("SAM2_CHECKPOINT_DIR"):
        cands.append(Path(os.environ["SAM2_CHECKPOINT_DIR"]) / name)
    spec = importlib.util.find_spec("sam2")
    if spec is not None and spec.origin:
        cands.append(Path(spec.origin).resolve().parents[1] / "checkpoints" / name)  # editable checkout
    cands += [Path.home() / "sam2_repo" / "checkpoints" / name, Path.home() / "checkpoints" / name]
    return next((c for c in cands if c.exists()), None)


def box_from_mask(coarse: np.ndarray, pad: int = 10) -> np.ndarray | None:
    ys, xs = np.nonzero(coarse > 127)
    if len(ys) == 0:
        return None
    h, w = coarse.shape[:2]
    return np.array([max(0, xs.min() - pad), max(0, ys.min() - pad), min(w - 1, xs.max() + pad),
                     min(h - 1, ys.max() + pad)], np.float32)


def run_sam(cams, frames, images_root: Path, coarse_root: Path, sam_root: Path, ext: str, model: str,
            checkpoint: Path, device: str) -> tuple[int, float]:
    """SAM 2 image predictor prompted with the bounding box of the coarse mask; writes
    ``sam_root/<cam>/<frame>.png`` (0/255). Existing files are reused. Returns (new, seconds)."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    todo = [(c, f) for c in cams for f in frames if not (sam_root / c / f"{f:06d}.png").exists()]
    if not todo:
        return 0, 0.0
    t0 = time.time()
    pred = SAM2ImagePredictor(build_sam2(SAM_CONFIGS[model], str(checkpoint), device=device))
    for c in cams:
        (sam_root / c).mkdir(parents=True, exist_ok=True)
    failed = 0
    for i, (c, f) in enumerate(todo, 1):
        for attempt in range(4):
            try:
                img = cv2.imread(str(images_root / c / f"{f:06d}.{ext}"))
                coarse = cv2.imread(str(coarse_root / c / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
                box = box_from_mask(coarse) if coarse is not None else None
                if img is None or box is None:
                    break
                with torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16,
                                                            enabled=device.startswith("cuda")):
                    pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    m, _, _ = pred.predict(box=box, multimask_output=False)
                cv2.imwrite(str(sam_root / c / f"{f:06d}.png"), (m[0] > 0).astype(np.uint8) * 255)
                break
            except (RuntimeError, MemoryError, cv2.error) as e:  # transient host / GPU memory pressure
                pred.reset_predictor()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                if attempt == 3:
                    failed += 1
                    print(f"  SAM 2 failed on {c}/{f:06d} ({str(e)[:80]}); refined without the person hint")
                else:
                    time.sleep(5.0 * (attempt + 1))
        if i % 100 == 0 or i == len(todo):
            print(f"  SAM 2: {i}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    del pred
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return len(todo) - failed, time.time() - t0


def pick_sam_device(requested: str, min_free_gb: float = 2.5) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free / 2 ** 30 >= min_free_gb:
            return "cuda"
        print(f"  GPU has only {free / 2 ** 30:.1f} GB free; SAM 2 runs on the CPU (slow)")
    return "cpu"


# ======================================================================================
# Workers
# ======================================================================================
_W: dict = {}


def _init(cfg: dict):
    _W.clear()
    _W.update(cfg)
    _W["params"] = RefineParams(**cfg["params"])


def _paths(cam: str, f: int):
    return (_W["images"] / cam / f"{f:06d}.{_W['ext']}", _W["coarse"] / cam / f"{f:06d}.png",
            (_W["depth"] / cam / f"{f:06d}") if _W["depth"] is not None else None)


def _do_chunk(task):
    """(camera, frames) -> per-frame stats; writes the refined masks."""
    cam, frames, all_frames = task
    p: RefineParams = _W["params"]
    geo = _W["geos"].get(cam)
    T = p.temporal
    fset = set(all_frames)
    need = sorted({g for f in frames for g in range(f - T, f + T + 1) if g in fset})
    cache: dict[int, tuple] = {}
    out_dir = _W["out"] / cam
    stats = []
    t_seg = 0.0
    for f in need:
        ip, mp, dp = _paths(cam, f)
        img = cv2.imread(str(ip))
        coarse = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or coarse is None:
            continue
        depth = read_depth(dp, _W["depth_scale"]) if dp is not None else None
        person = None
        if _W.get("sam") is not None:
            person = cv2.imread(str(_W["sam"] / cam / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
        t0 = time.perf_counter()
        fg, info = segment_frame(img, coarse, depth, geo, p, person)
        t_seg += time.perf_counter() - t0
        cache[f] = (fg, info.get("sure_fg"), info)
    for f in frames:
        if f not in cache:
            continue
        ip, mp, _ = _paths(cam, f)
        img = cv2.imread(str(ip))
        coarse = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        window = [cache[g][0] for g in range(f - T, f + T + 1) if g in cache]
        if len(window) % 2 == 0:  # sequence edge: centre frame only
            window = [cache[f][0]]
        prob = temporal_vote(window) if len(window) > 1 else cache[f][0].astype(np.float32)
        out = finish_mask(prob, img, coarse, cache[f][1], p, cache[f][2].get("allowed"))
        cv2.imwrite(str(out_dir / f"{f:06d}.png"), out)
        info = cache[f][2]
        stats.append({"frame": f, "coarse_frac": float((coarse > 127).mean()), "refined_frac": float((out > 127).mean()),
                      "sure_fg": info.get("sure_fg_frac_of_coarse"), "depth_bg": info.get("depth_bg_frac_of_coarse"),
                      "person": info.get("person_frac_of_coarse"), "depth": info.get("depth", False)})
    return cam, stats, t_seg


# ======================================================================================
# QA / metrics
# ======================================================================================
def qa_sheet(cams, images_root: Path, coarse_root: Path, refined_root: Path, frames, path: Path, ext: str = "jpg",
             tile_h: int = 640):
    """Image with background dimmed by the refined mask; coarse outline red, refined green."""
    rows = []
    for f in frames:
        tiles = []
        for c in cams:
            img = cv2.imread(str(images_root / c / f"{f:06d}.{ext}"))
            m0 = cv2.imread(str(coarse_root / c / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
            m1 = cv2.imread(str(refined_root / c / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None or m0 is None or m1 is None:
                continue
            a = (m1.astype(np.float32) / 255.0)[..., None]
            dim = (img * 0.25).astype(np.uint8)
            dim[..., 0] = np.clip(dim[..., 0].astype(int) + 60, 0, 255)
            comp = (img * a + dim * (1 - a)).astype(np.uint8)
            for m, col in ((m0, (0, 0, 255)), (m1, (0, 255, 0))):
                cnts, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(comp, cnts, -1, col, 2)
            ys, xs = np.nonzero(m0 > 127)
            if len(ys):  # crop to the coarse mask with a margin, keep aspect
                pad = 40
                y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
                x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
                comp = comp[y0:y1, x0:x1]
            cv2.putText(comp, f"{c} f{f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            s = tile_h / comp.shape[0]
            tiles.append(cv2.resize(comp, (max(1, int(comp.shape[1] * s)), tile_h), interpolation=cv2.INTER_AREA))
        if tiles:
            rows.append(np.hstack(tiles))
    if rows:
        w = max(r.shape[1] for r in rows)
        cv2.imwrite(str(path), np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT)
                                          for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 127, b > 127
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def temporal_iou(root: Path, cam: str, frames: list[int]) -> list[float]:
    """IoU between consecutive frames (only pairs f, f+1 that are both in ``frames``)."""
    out = []
    prev_f, prev = None, None
    for f in frames:
        m = cv2.imread(str(root / cam / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            prev_f, prev = None, None
            continue
        if prev is not None and f == prev_f + 1:
            out.append(iou(prev, m))
        prev_f, prev = f, m
    return out


# ======================================================================================
# CLI
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    d = RefineParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--images-dir", default="images")
    ap.add_argument("--image-ext", default="jpg")
    ap.add_argument("--masks-dir", default="masks", help="coarse masks (input)")
    ap.add_argument("--output-dir", default=None, help="default: <data-dir>/masks_refined")
    ap.add_argument("--report-dir", default=None, help="default: <data-dir>/mask_report")
    ap.add_argument("--reference-json", default="reference_export.json")
    ap.add_argument("--depth-dir", default="depth_registered",
                    help="registered depth (uint16 PNG or float32 .npy); missing directory = colour only")
    ap.add_argument("--depth-scale", type=float, default=0.001, help="metres per PNG unit")
    ap.add_argument("--no-depth", action="store_true", help="ignore depth even if present")
    ap.add_argument("--ground-json", default="mask_report/masks.json",
                    help="depth_masks report with the floor plane (relative to --data-dir); "
                         "missing = fit the floor from the registered depth")
    ap.add_argument("--cameras", default=None, help="comma-separated camera folders (default: all with masks)")
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None, help="inclusive")
    ap.add_argument("--frames", default=None, help="explicit list, e.g. 1-31,100,200 (overrides start/end)")
    ap.add_argument("--qa-frames", default=None, help="frames on the QA sheet (default: 5 spread over the run)")
    ap.add_argument("--qa-only", action="store_true", help="only process the QA frames (fast tuning)")
    for k, v in asdict(d).items():
        if k == "use_depth":
            continue
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--sam", choices=("auto", "on", "off"), default="auto",
                    help="SAM 2 person masks as a sure-foreground hint (auto: when sam2 and a checkpoint are found)")
    ap.add_argument("--sam-model", choices=sorted(SAM_CONFIGS), default="large")
    ap.add_argument("--sam-checkpoint", default=None, help="default: search $SAM2_CHECKPOINT_DIR, the sam2 checkout, "
                    "~/sam2_repo/checkpoints, ~/checkpoints")
    ap.add_argument("--sam-device", default="auto", help="auto: cuda if >= 2.5 GB free, else cpu")
    ap.add_argument("--sam-dir", default=None, help="cache of SAM 2 person masks (default: <data-dir>/masks_sam); "
                    "existing files are reused")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--chunk", type=int, default=24, help="frames per worker task")
    return ap


def parse_frames(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data = Path(args.data_dir)
    images_root = data / args.images_dir
    coarse_root = data / args.masks_dir
    out_root = Path(args.output_dir) if args.output_dir else data / "masks_refined"
    report_dir = Path(args.report_dir) if args.report_dir else data / "mask_report"
    if out_root.resolve() == coarse_root.resolve():
        raise SystemExit("--output-dir must differ from the coarse masks directory")
    report_dir.mkdir(parents=True, exist_ok=True)
    p = RefineParams(**{k: getattr(args, k) for k in asdict(RefineParams()) if k != "use_depth"})
    p.use_depth = not args.no_depth

    cams = sorted(d.name for d in coarse_root.iterdir() if d.is_dir())
    if args.cameras:
        cams = [c for c in args.cameras.split(",") if c in cams]
    if not cams:
        raise SystemExit(f"no camera folders in {coarse_root}")
    nums = None
    for c in cams:
        s = {int(q.stem) for q in (coarse_root / c).glob("*.png") if q.stem.isdigit()}
        s &= {int(q.stem) for q in (images_root / c).glob(f"*.{args.image_ext}") if q.stem.isdigit()}
        nums = s if nums is None else nums & s
    if not nums:
        raise SystemExit("no frames with both an image and a coarse mask")
    if args.frames:
        frames = [f for f in parse_frames(args.frames) if f in nums]
    else:
        lo = min(nums) if args.frame_start is None else args.frame_start
        hi = max(nums) if args.frame_end is None else args.frame_end
        frames = [f for f in sorted(nums) if lo <= f <= hi]
    qa = parse_frames(args.qa_frames) if args.qa_frames else \
        sorted({frames[int(round(i))] for i in np.linspace(0, len(frames) - 1, 5)})
    qa = [f for f in qa if f in nums]
    if args.qa_only:
        frames = qa

    depth_root = data / args.depth_dir
    if args.no_depth or not depth_root.is_dir():
        depth_root = None
    gj = Path(args.ground_json)
    gj = gj if gj.is_absolute() else data / gj
    geos, ginfo = build_geometries(data, args.reference_json, depth_root, args.depth_scale, gj, frames, cams)
    print(f"{len(cams)} cameras, {len(frames)} frames, depth: {'yes' if depth_root else 'no'}"
          + (f", floor: {ginfo['floor']}, stage: {ginfo['stage']}" if ginfo else ""))

    sam_root, sam_info = None, {"used": False}
    if args.sam != "off":
        ckpt = find_sam_checkpoint(args.sam_model, args.sam_checkpoint) \
            if importlib.util.find_spec("sam2") is not None else None
        if ckpt is None:
            if args.sam == "on":
                raise SystemExit("--sam on: the sam2 package or the checkpoint was not found (see README-refine-masks.md)")
            print("SAM 2 not available: colour + depth only")
        else:
            sam_root = Path(args.sam_dir) if args.sam_dir else data / "masks_sam"
            # In a child process, so the torch / CUDA memory is released before the CPU workers start.
            with ProcessPoolExecutor(max_workers=1) as pool:
                dev = pool.submit(pick_sam_device, args.sam_device).result()
                print(f"SAM 2 ({args.sam_model}, {dev}) person masks -> {sam_root}")
                n_new, t_sam = pool.submit(run_sam, cams, frames, images_root, coarse_root, sam_root, args.image_ext,
                                           args.sam_model, ckpt, dev).result()
            sam_info = {"used": True, "model": args.sam_model, "checkpoint": str(ckpt), "device": dev,
                        "dir": str(sam_root), "new_masks": n_new, "seconds": t_sam}
            print(f"  SAM 2: {n_new} new masks in {t_sam:.0f} s")

    for c in cams:
        (out_root / c).mkdir(parents=True, exist_ok=True)
    # Chunks of consecutive frames; the temporal vote reads up to `temporal` frames beyond each chunk.
    tasks = []
    for c in cams:
        for i in range(0, len(frames), args.chunk):
            tasks.append((c, frames[i:i + args.chunk], frames))
    cfg = {"images": images_root, "coarse": coarse_root, "depth": depth_root, "depth_scale": args.depth_scale,
           "ext": args.image_ext, "out": out_root, "geos": geos, "params": asdict(p), "sam": sam_root}
    t0 = time.time()
    per_cam: dict[str, list] = {c: [] for c in cams}
    seg_time = 0.0
    with contextlib.ExitStack() as stack:
        if args.workers > 1:
            pool = stack.enter_context(ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                                           initargs=(cfg,)))
            results = pool.map(_do_chunk, tasks)
        else:  # in-process (also used by the tests)
            _init(cfg)
            results = map(_do_chunk, tasks)
        for i, (c, stats, ts) in enumerate(results, 1):
            per_cam[c].extend(stats)
            seg_time += ts
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} tasks ({time.time() - t0:.0f} s)", flush=True)
    wall = time.time() - t0

    qa_sheet(cams, images_root, coarse_root, out_root, qa, report_dir / "refine_sheet.jpg", args.image_ext)
    summary = {}
    for c in cams:
        rows = sorted(per_cam[c], key=lambda r: r["frame"])
        if not rows:
            continue
        cf = np.array([r["coarse_frac"] for r in rows])
        rf = np.array([r["refined_frac"] for r in rows])
        ratio = rf / np.maximum(cf, 1e-9)
        fr = [r["frame"] for r in rows]
        ti0, ti1 = temporal_iou(coarse_root, c, fr), temporal_iou(out_root, c, fr)
        jumps = np.abs(np.diff(rf)) / np.maximum(rf[:-1], 1e-6) if len(rf) > 1 else np.zeros(0)
        consec = np.diff(fr) == 1 if len(fr) > 1 else np.zeros(0, bool)
        flagged = [int(fr[i + 1]) for i in np.nonzero((jumps > 0.10) & consec)[0]]
        summary[c] = {
            "coarse_fg_fraction_median": float(np.median(cf)), "refined_fg_fraction_median": float(np.median(rf)),
            "area_ratio_median": float(np.median(ratio)), "area_ratio_min": float(ratio.min()),
            "area_ratio_max": float(ratio.max()),
            "temporal_iou_coarse_mean": float(np.mean(ti0)) if ti0 else None,
            "temporal_iou_refined_mean": float(np.mean(ti1)) if ti1 else None,
            "temporal_iou_refined_min": float(np.min(ti1)) if ti1 else None,
            "frames_with_area_jump_gt_10pct": flagged[:50],
        }
        s = summary[c]
        print(f"  {c}: area {s['area_ratio_median'] * 100:.0f}% of coarse (min {s['area_ratio_min'] * 100:.0f}, "
              f"max {s['area_ratio_max'] * 100:.0f}); temporal IoU coarse "
              f"{s['temporal_iou_coarse_mean'] or float('nan'):.3f} -> refined "
              f"{s['temporal_iou_refined_mean'] or float('nan'):.3f} (min {s['temporal_iou_refined_min'] or float('nan'):.3f}); "
              f"{len(flagged)} jumps > 10%")
    n_img = sum(len(v) for v in per_cam.values())
    print(f"{n_img} masks in {wall:.0f} s ({wall / max(1, n_img) * 1000:.0f} ms per mask wall, "
          f"{seg_time / max(1, n_img) * 1000:.0f} ms per mask segmentation CPU)")
    report = {"params": asdict(p), "depth": str(depth_root) if depth_root else None, "ground": ginfo, "sam": sam_info,
              "coarse_dir": str(coarse_root), "output_dir": str(out_root),
              "frames": frames if len(frames) < 64 else [frames[0], frames[-1], len(frames)],
              "qa_frames": qa, "qa_only": args.qa_only, "wall_seconds": wall, "masks": n_img, "cameras": summary}
    (report_dir / "refine.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Refined masks: {out_root}  QA: {report_dir / 'refine_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
