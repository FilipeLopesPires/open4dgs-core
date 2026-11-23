#!/usr/bin/env python3
"""Tests for prep/mask/depth_masks.py on a synthetic, analytically ray-cast depth rig.

Scene (canonical frame: z up, floor at z = 0): six upright pinhole cameras on a ring
(radius 1.5 m, height 1.7 m) looking at (0, 0, 0.9); a "person" (vertical cylinder,
r = 0.3 m, 1.6 m tall) standing on the floor slightly off the ring centre; optional floor
cable, stray clump, occluder pole and floor noise.  Depth is z-depth along each camera's
optical axis, written as registered depth (uint16 mm PNG or float32 metres .npy).
"""
import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "depth_masks.py"
SPEC = importlib.util.spec_from_file_location("depth_masks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

# ------------------------------------------------------------------------------------
# Synthetic rig
# ------------------------------------------------------------------------------------
W, H, FOCAL = 240, 320, 250.0  # portrait images
N_CAMS, RING_R, CAM_H, TARGET_H = 6, 1.5, 1.7, 0.9
FRAME = 1

LBL_NONE, LBL_FLOOR, LBL_PERSON, LBL_CABLE, LBL_CLUMP, LBL_POLE = range(6)
PX, PY, PR, P_TOP = 0.08, -0.06, 0.30, 1.60  # person axis, radius, height
PERSON = (LBL_PERSON, "cyl", (PX, PY, PR, 0.0, P_TOP))
# The cable is 5 cm high (above the 2.5 cm floor margin) so that the *column-support* rule,
# not the floor margin, is what has to remove it (checked against --low-band 0).
CABLE = (LBL_CABLE, "box", ((PX + PR - 0.02, PY - 0.015, 0.0), (PX + 1.3, PY + 0.015, 0.05)))
# Hanging clump 0.55 m from the person axis: a 0.19 m air gap (> 2 cluster voxels).
CLUMP = (LBL_CLUMP, "cyl", (PX, PY + 0.55, 0.06, 0.75, 0.90))
# Pole between camera 0001 (at (1.5, 0, 1.7)) and the person, outside the stage cylinder.
POLE = (LBL_POLE, "cyl", (1.0, -0.02, 0.06, 0.0, 2.0))
FLOOR_RADIUS = 3.5

_DEF = MODULE.Params.__dataclass_fields__
SPLAT, CLOSE, DILATE = (_DEF[k].default for k in ("splat_px", "close_px", "dilate_px"))
EDGE_PX = SPLAT + DILATE  # expected outward growth of a mask w.r.t. the true silhouette


def ring_cameras(n=N_CAMS):
    """World->camera (R, t) of upright cameras: image y axis points down (towards -z)."""
    cams, target = [], np.array([0.0, 0.0, TARGET_H])
    for k in range(n):
        a = 2 * math.pi * k / n
        C = np.array([RING_R * math.cos(a), RING_R * math.sin(a), CAM_H])
        fwd = (target - C) / np.linalg.norm(target - C)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        R = np.stack([right, np.cross(fwd, right), fwd])
        K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1.0]])
        cams.append({"folder": f"{k + 1:04d}", "K": K, "R": R, "t": -R @ C, "C": C})
    return cams


def _hit_cylinder(o, d, cx, cy, r, z0, z1):
    ox, oy = o[0] - cx, o[1] - cy
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        ts = (-b - np.sqrt(disc)) / (2 * a)
        z = o[2] + ts * dz
        t = np.where((disc >= 0) & (a > 1e-12) & (ts > 0) & (z >= z0) & (z <= z1), ts, np.inf)
        for zc in (z0, z1):  # caps
            tc = (zc - o[2]) / dz
            qx, qy = o[0] + tc * dx - cx, o[1] + tc * dy - cy
            t = np.where((tc > 0) & (qx * qx + qy * qy <= r * r) & (tc < t), tc, t)
    return t


def _hit_box(o, d, lo, hi):
    with np.errstate(invalid="ignore", divide="ignore"):
        inv = 1.0 / d
        t0 = (np.asarray(lo) - o) * inv
        t1 = (np.asarray(hi) - o) * inv
        tmin = np.minimum(t0, t1).max(axis=-1)
        tmax = np.maximum(t0, t1).min(axis=-1)
        return np.where((tmin <= tmax) & (tmin > 0), tmin, np.inf)


def render(cam, objects, floor=True, size=(W, H)):
    """Ray-cast z-depth (0 = no hit), object labels and world hit points.

    ``size`` is the depth resolution; depth pixel j samples image coordinate
    (j + 0.5) * W / w - 0.5, the convention RegisteredDepthSource uses for scaled depth."""
    w, h = size
    u, v = np.meshgrid((np.arange(w) + 0.5) * W / w - 0.5, (np.arange(h) + 0.5) * H / h - 0.5)
    K = cam["K"]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d = rays @ cam["R"]  # world directions with unit camera z -> ray parameter == z-depth
    o = cam["C"]
    depth = np.full((h, w), np.inf)
    label = np.zeros((h, w), np.uint8)
    if floor:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -o[2] / d[..., 2]
        p = o + d * np.where(np.isfinite(t), t, 0)[..., None]
        t = np.where((d[..., 2] < 0) & (np.hypot(p[..., 0], p[..., 1]) < FLOOR_RADIUS), t, np.inf)
        label[t < depth], depth = LBL_FLOOR, np.minimum(depth, t)
    for lbl, kind, prm in objects:
        t = _hit_cylinder(o, d, *prm) if kind == "cyl" else _hit_box(o, d, *prm)
        closer = t < depth
        depth[closer], label[closer] = t[closer], lbl
    hit = o + d * np.where(np.isfinite(depth), depth, 0)[..., None]
    depth[~np.isfinite(depth)] = 0.0
    return depth, label, hit


def add_floor_noise(depth, label, seed=0, frac=0.03, max_pull=0.04):
    """Pull a few floor samples towards the camera: spurious points 0-7 cm above the floor."""
    rng = np.random.default_rng(seed)
    pick = (label == LBL_FLOOR) & (rng.random(depth.shape) < frac)
    out = depth.copy()
    out[pick] *= 1.0 - rng.uniform(0.0, max_pull, int(pick.sum()))
    return out


def random_rigid(seed=3):
    """A rotation not aligned with any axis, and a translation."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    Q, _ = cv2.Rodrigues(axis * 1.1)
    return Q, np.array([2.3, -4.1, 0.7])


def write_dataset(root, cams, depths, frames=(FRAME,), fmt="png", transform=None):
    """reference_export.json + depth_registered/<cam>/<frame>.(png|npy) + dummy images.

    ``transform`` = (Q, s) expresses the world as X' = Q X + s (depth is unchanged)."""
    root = Path(root)
    entries = []
    for cam, dep in zip(cams, depths):
        R, t = cam["R"], cam["t"]
        if transform is not None:
            Q, s = transform
            R, t = R @ Q.T, t - R @ Q.T @ s
        K = cam["K"]
        entries.append({"folder": cam["folder"], "camera_model_name": "PINHOLE",
                        "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], "width": W, "height": H,
                        "R": R.tolist(), "t": t.tolist()})
        (root / "depth_registered" / cam["folder"]).mkdir(parents=True, exist_ok=True)
        (root / "images" / cam["folder"]).mkdir(parents=True, exist_ok=True)
        for f in frames:
            base = root / "depth_registered" / cam["folder"] / f"{f:06d}"
            if fmt == "npy":
                np.save(base.with_suffix(".npy"), dep.astype(np.float32))
            else:
                cv2.imwrite(str(base.with_suffix(".png")), np.round(dep * 1000).astype(np.uint16))
            cv2.imwrite(str(root / "images" / cam["folder"] / f"{f:06d}.jpg"),
                        np.full((H, W, 3), 90, np.uint8))
    (root / "reference_export.json").write_text(json.dumps({"cameras": entries}), encoding="utf-8")
    return root


def run_pipeline(data_dir, ground=None, no_floor=False, radius=None, frame=FRAME, **overrides):
    """In-process equivalent of main() with CLI defaults, for one frame."""
    cams = MODULE.load_cameras(Path(data_dir))
    src = MODULE.make_source("registered", Path(data_dir), cams, None, 0.001)
    if ground is None:
        ground = MODULE.estimate_ground(cams, src, [frame], no_floor=no_floor)
    # same as main(): only points above the low band (default 0.12 m) locate the subject
    min_h = max(0.025, overrides.get("low_band", 0.12)) if ground.has_floor else 0.025
    center, r_est = MODULE.estimate_stage(cams, src, [frame], ground, min_h, 1.2)
    radius = radius if radius is not None else r_est + 0.25
    bias = ()
    if ground.has_floor:
        bias = tuple(MODULE.floor_bias(cams, src, [frame], ground, center, radius + 0.1, radius + 0.8))
    p = MODULE.Params(center=center, radius=radius, floor_margin=0.025 if ground.has_floor else -1e9,
                      top=2.3 if ground.has_floor else 1e9, floor_bias=bias, **overrides)
    world = [MODULE.to_world(c, src.points_cam(i, frame)) for i, c in enumerate(cams)]
    masks, _ = MODULE.build_masks(cams, world, ground, p)
    return [m > 127 for m in masks], ground, p, cams, world


def iou(a, b):
    return np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum())


def grow(binary, px):
    return cv2.dilate(binary.astype(np.uint8), MODULE._disk(px)) > 0


def dist_from(binary):
    """Pixel distance to the nearest True pixel of ``binary``."""
    return cv2.distanceTransform((~binary).astype(np.uint8), cv2.DIST_L2, 5)


def angle_deg(a, b):
    return math.degrees(math.acos(np.clip(abs(np.dot(a, b)) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


class DepthMasksSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.cams = ring_cameras()
        cls.base = [render(c, [PERSON]) for c in cls.cams]
        cls.base_dir = write_dataset(cls.tmp / "base", cls.cams, [r[0] for r in cls.base])
        cls.base_masks, cls.base_ground, cls.base_params, _, _ = run_pipeline(cls.base_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def scene(self, name, objects, noise=False, floor=True, **kw):
        renders = [render(c, objects, floor=floor) for c in self.cams]
        depths = [add_floor_noise(d, lab, seed=i) if noise else d for i, (d, lab, _) in enumerate(renders)]
        return write_dataset(self.tmp / name, self.cams, depths, **kw), renders

    # 1 ------------------------------------------------------------------------------
    def test_world_frame_is_source_agnostic(self):
        Q, s = random_rigid()
        for k in range(3):  # truly arbitrary: no column of Q near a coordinate axis
            self.assertLess(np.abs(Q[:, k]).max(), 0.97)
        data = write_dataset(self.tmp / "rigid", self.cams, [r[0] for r in self.base], transform=(Q, s))
        masks, ground, _, cams, _ = run_pipeline(data)
        self.assertLess(angle_deg(ground.up, Q @ [0, 0, 1.0]), 2.0)
        self.assertGreater(ground.up @ (Q @ [0, 0, 1.0]), 0)  # oriented away from the floor
        for c in cams:
            self.assertAlmostEqual(float(ground.up @ c.center + ground.d), CAM_H, delta=0.03)
        for ref, m in zip(self.base_masks, masks):
            self.assertGreater(iou(ref, m), 0.98)
        # untransformed reference: z is up
        self.assertLess(angle_deg(self.base_ground.up, [0, 0, 1.0]), 2.0)

    # 2 ------------------------------------------------------------------------------
    def test_masks_match_analytic_silhouette(self):
        for (_, label, _), m in zip(self.base, self.base_masks):
            sil = label == LBL_PERSON
            expected = grow(grow(sil, SPLAT), DILATE)
            self.assertGreater(iou(m, expected), 0.85)
            self.assertGreater(m[sil].mean(), 0.99)  # the whole subject is inside
            self.assertEqual(int((m & (dist_from(sil) > EDGE_PX + CLOSE)).sum()), 0)

    # 3 ------------------------------------------------------------------------------
    def test_floor_cable_and_noise_removed_feet_kept(self):
        data, renders = self.scene("cable", [PERSON, CABLE], noise=True)
        # explicit radius: keep the cable inside the cylinder so the column rule is what removes it
        masks, _, p, _, _ = run_pipeline(data, ground=self.base_ground, radius=1.2)
        loose, _, p_loose, _, _ = run_pipeline(data, ground=self.base_ground, radius=1.2, low_band=0.0)
        cable_in, cable_cov, cable_cov_loose, feet_cov, stray = 0, [], [], [], 0
        for (_, label, hit), m, ml in zip(renders, masks, loose):
            radial = np.hypot(hit[..., 0] - PX, hit[..., 1] - PY)
            stage_r = min(p.radius, p_loose.radius) - 0.05
            sil = label == LBL_PERSON
            # Cable inside the stage, away from the supported columns at the feet (world) and
            # beyond the mask's legitimate growth around the silhouette (image).
            far_cable = ((label == LBL_CABLE) & (radial > PR + 0.15) & (radial < stage_r)
                         & (dist_from(sil) > EDGE_PX + CLOSE))
            cable_in += int(far_cable.sum())
            cable_cov += list(m[far_cable])
            cable_cov_loose += list(ml[far_cable])
            feet = sil & (hit[..., 2] < 0.10)
            feet_cov += list(m[feet])
            near = (label != LBL_NONE) & (radial < PR + 0.15)  # person + floor/cable stub at its feet
            stray += int((m & ~grow(near, EDGE_PX + CLOSE)).sum())
        self.assertGreater(cable_in, 200, "cable must be visible inside the stage cylinder")
        self.assertLess(np.mean(cable_cov), 0.05)
        self.assertGreater(np.mean(cable_cov_loose), 0.5, "without column support the cable is kept")
        self.assertGreater(len(feet_cov), 500)
        self.assertGreater(np.mean(feet_cov), 0.95)
        self.assertLess(stray, 0.001 * W * H * N_CAMS)

    def test_floor_clutter_does_not_inflate_stage(self):
        """A raised cable next to the subject must not stretch the estimated stage radius."""
        data, _ = self.scene("cable_stage", [PERSON, CABLE])
        _, _, p_cable, _, _ = run_pipeline(data, ground=self.base_ground)
        self.assertAlmostEqual(p_cable.radius, self.base_params.radius, delta=0.05)

    # 4 ------------------------------------------------------------------------------
    def test_stray_clump_removed_by_main_cluster(self):
        data, renders = self.scene("clump", [PERSON, CLUMP])
        masks, _, _, _, _ = run_pipeline(data, ground=self.base_ground, radius=0.85)
        kept, _, _, _, _ = run_pipeline(data, ground=self.base_ground, radius=0.85, cluster_voxel=0.0)
        cov, cov_kept = [], []
        for (_, label, _), m, mk in zip(renders, masks, kept):
            far = (label == LBL_CLUMP) & (dist_from(label == LBL_PERSON) > EDGE_PX + CLOSE)
            cov += list(m[far])
            cov_kept += list(mk[far])
        self.assertGreater(len(cov), 200)
        self.assertLess(np.mean(cov), 0.05)
        self.assertGreater(np.mean(cov_kept), 0.8)

    def test_main_cluster_direct(self):
        rng = np.random.default_rng(0)
        big = rng.uniform([0, 0, 0], [0.5, 0.5, 1.5], (5000, 3))
        small = rng.uniform([1.0, 1.0, 1.0], [1.1, 1.1, 1.1], (100, 3))
        keep = MODULE.main_cluster(np.concatenate([big, small]).astype(np.float32), 0.05, 0.15)
        self.assertTrue(keep[:5000].all())
        self.assertFalse(keep[5000:].any())

    # 5 ------------------------------------------------------------------------------
    def test_depth_dropout_filled_by_other_views(self):
        ci = 2
        depths = [r[0].copy() for r in self.base]
        ys, xs = np.nonzero(self.base[ci][1] == LBL_PERSON)
        cy, cx = int(ys.mean()), int(xs.mean())
        hole = (slice(cy - 24, cy + 24), slice(cx - 24, cx + 24))
        self.assertTrue((self.base[ci][1][hole] == LBL_PERSON).all())
        depths[ci][hole] = 0
        data = write_dataset(self.tmp / "dropout", self.cams, depths)
        masks, ground, p, cams, world = run_pipeline(data, ground=self.base_ground)
        core = (slice(cy - 8, cy + 8), slice(cx - 8, cx + 8))
        self.assertTrue(masks[ci][core].all(), "multi-view fusion fills the dropout")
        # Control: the same camera alone cannot fill it (morphology / bounded fill do not).
        alone = [w if i == ci else np.zeros((0, 3), np.float32) for i, w in enumerate(world)]
        single, _ = MODULE.build_masks(cams, alone, ground, p)
        self.assertLess((single[ci][core] > 127).mean(), 0.5)

    def test_fill_holes_is_bounded(self):
        img = np.zeros((200, 200), np.uint8)
        cv2.circle(img, (100, 100), 80, 1, -1)
        cv2.circle(img, (100, 100), 30, 0, -1)  # large enclosed region (~2800 px)
        cv2.circle(img, (100, 35), 4, 0, -1)  # small dropout (~50 px)
        img[0:10, 90:110] = 1
        img[0:5, 98:102] = 0  # small notch open to the border: never filled
        out = MODULE.fill_holes(img, max_area=200)
        self.assertEqual(out[35, 100], 1)
        self.assertEqual(out[100, 100], 0)
        self.assertEqual(out[0:5, 98:102].max(), 0)
        np.testing.assert_array_equal(out[img > 0], 1)  # never removes foreground
        self.assertEqual(MODULE.fill_holes(img, max_area=5000)[100, 100], 1)

    # 6 ------------------------------------------------------------------------------
    def test_occluder_carves_mask(self):
        data, renders = self.scene("pole", [PERSON, POLE])
        masks, _, p, _, _ = run_pipeline(data, ground=self.base_ground)
        pole_r = math.hypot(POLE[2][0] - p.center[0], POLE[2][1] - p.center[1])
        self.assertGreater(pole_r - POLE[2][2], p.radius, "pole must be outside the stage cylinder")
        ci = 0  # camera 0001 looks at the person through the pole
        person_only = render(self.cams[ci], [PERSON])[1] == LBL_PERSON
        label = renders[ci][1]
        occluded = (label == LBL_POLE) & person_only
        margin = EDGE_PX + 3
        core = cv2.erode(occluded.astype(np.uint8), np.ones((1, 2 * margin + 1), np.uint8)) > 0
        self.assertGreater(core.sum(), 500)
        self.assertLess(masks[ci][core].mean(), 0.05)
        visible = cv2.erode((label == LBL_PERSON).astype(np.uint8), MODULE._disk(2)) > 0
        self.assertGreater(masks[ci][visible].mean(), 0.97)
        # Other cameras still see the whole subject.
        for (_, lab, _), m in list(zip(renders, masks))[1:]:
            self.assertGreater(m[lab == LBL_PERSON].mean(), 0.97)

    # 7 ------------------------------------------------------------------------------
    def test_half_resolution_depth(self):
        depths = [render(c, [PERSON], size=(W // 2, H // 2))[0] for c in self.cams]
        data = write_dataset(self.tmp / "half", self.cams, depths)
        masks, ground, _, _, _ = run_pipeline(data)
        self.assertLess(angle_deg(ground.up, [0, 0, 1.0]), 2.0)
        for ref, m in zip(self.base_masks, masks):
            self.assertEqual(m.shape, (H, W))
            self.assertGreater(iou(ref, m), 0.95)

    def test_npy_metres_matches_png_millimetres(self):
        data = write_dataset(self.tmp / "npy", self.cams, [r[0] for r in self.base], fmt="npy")
        self.assertFalse(list((data / "depth_registered").rglob("*.png")))
        masks, _, _, _, _ = run_pipeline(data, ground=self.base_ground)
        for ref, m in zip(self.base_masks, masks):
            self.assertGreater(iou(ref, m), 0.98)

    def test_depth_with_wrong_aspect_is_rejected(self):
        data = write_dataset(self.tmp / "aspect", self.cams[:1], [np.ones((100, 100))])
        cams = MODULE.load_cameras(data)
        src = MODULE.RegisteredDepthSource(data, cams)
        with self.assertRaises(ValueError):
            src.points_cam(0, FRAME)

    # 8 ------------------------------------------------------------------------------
    def test_no_floor_mode(self):
        data, renders = self.scene("nofloor", [PERSON], floor=False)
        masks, ground, p, _, _ = run_pipeline(data, no_floor=True)
        self.assertFalse(ground.has_floor)
        self.assertLess(angle_deg(ground.up, [0, 0, 1.0]), 1.0)  # mean camera image-up
        for (_, label, _), m in zip(renders, masks):
            sil = label == LBL_PERSON
            self.assertGreater(m[sil].mean(), 0.98)
            self.assertGreater(iou(m, grow(grow(sil, SPLAT), DILATE)), 0.85)

    def test_camera_up_hint(self):
        cams = MODULE.load_cameras(self.base_dir)
        self.assertLess(angle_deg(MODULE.camera_up_hint(cams), [0, 0, 1.0]), 0.5)
        self.assertGreater(MODULE.camera_up_hint(cams)[2], 0)

    # 9 ------------------------------------------------------------------------------
    def test_intrinsics_models(self):
        K, dist = MODULE._intrinsics("PINHOLE", [500, 510, 320, 240])
        np.testing.assert_allclose(K, [[500, 0, 320], [0, 510, 240], [0, 0, 1]])
        np.testing.assert_array_equal(dist, np.zeros(4))
        K, dist = MODULE._intrinsics("OPENCV", [500, 510, 320, 240, 0.1, -0.02, 0.001, 0.002])
        np.testing.assert_allclose(K, [[500, 0, 320], [0, 510, 240], [0, 0, 1]])
        np.testing.assert_allclose(dist, [0.1, -0.02, 0.001, 0.002])
        K, dist = MODULE._intrinsics("SIMPLE_RADIAL", [600, 320, 240, 0.05])
        np.testing.assert_allclose(K, [[600, 0, 320], [0, 600, 240], [0, 0, 1]])
        np.testing.assert_allclose(dist, [0.05, 0, 0, 0])
        self.assertEqual(dist.dtype, np.float64)
        for model in ("OPENCV_FISHEYE", "THIN_PRISM_FISHEYE", "BOGUS"):
            with self.assertRaises(ValueError):
                MODULE._intrinsics(model, [500, 500, 320, 240, 0, 0, 0, 0])

    # 10 -----------------------------------------------------------------------------
    def test_cli_end_to_end(self):
        frames = (1, 2)
        data = write_dataset(self.tmp / "cli", self.cams, [r[0] for r in self.base], frames=frames)
        # Spawned worker processes import the module by name.
        sys.path.insert(0, str(SCRIPT.parent))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = MODULE.main(["--data-dir", str(data), "--workers", "1"])
        finally:
            sys.path.remove(str(SCRIPT.parent))
        self.assertEqual(rc, 0)
        for (_, label, _), c in zip(self.base, self.cams):
            for f in frames:
                m = cv2.imread(str(data / "masks" / c["folder"] / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
                self.assertIsNotNone(m)
                self.assertEqual(m.shape, (H, W))
                self.assertGreater((m[label == LBL_PERSON] > 127).mean(), 0.99)
        report = json.loads((data / "mask_report" / "masks.json").read_text(encoding="utf-8"))
        self.assertEqual(report["depth_source"], "registered")
        self.assertEqual(report["frames"], [1, 2])
        self.assertEqual(sorted(report["cameras"]), [c["folder"] for c in self.cams])
        self.assertTrue((data / "mask_report" / "mask_sheet.jpg").exists())


if __name__ == "__main__":
    unittest.main()
