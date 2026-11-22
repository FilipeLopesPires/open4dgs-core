#!/usr/bin/env python3
"""Tests for prep/extract/k4a_depth_offsets.py on a synthetic, analytically ray-cast rig.

Scene (+Y up, metres): six upright pinhole cameras on a ring (radius 1.5 m, 60 deg apart,
height 1.2 m) looking at an ellipsoid-plus-sphere "subject".  Depth is z-depth of each
colour camera at half resolution (uint16 mm PNG, pixel centre full = (half + 0.5) * 2 - 0.5,
the registered-depth contract).  A known per-camera bias is injected: measured = true - o,
so the correction to add is o.
"""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "k4a_depth_offsets.py"
SPEC = importlib.util.spec_from_file_location("k4a_depth_offsets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

W, H, FOCAL = 480, 640, 700.0  # colour image; depth is W/2 x H/2 (~3 mm per depth pixel)
N_CAMS, RING_R, CAM_H = 6, 1.5, 1.2
TARGET = np.array([0.0, 0.85, 0.0])
BODY_C, BODY_AX = np.array([0.0, 0.75, 0.0]), np.array([0.36, 0.40, 0.30])  # ellipsoid semi-axes
HEAD_C, HEAD_R = np.array([0.02, 1.25, 0.03]), 0.13
TRUE_OFFSETS = np.array([0.006, -0.004, 0.010, -0.012, 0.003, -0.001])  # metres to add


def look_at(center, target):
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ center


def make_cams():
    cams = []
    for i in range(N_CAMS):
        a = 2 * np.pi * i / N_CAMS + 0.3
        C = np.array([RING_R * np.cos(a), CAM_H, RING_R * np.sin(a)])
        R, t = look_at(C, TARGET)
        cams.append({"folder": f"{i + 1:04d}", "R": R, "t": t, "C": C})
    return cams


def ray_ellipsoid(C, d, centre, axes):
    s = 1.0 / axes
    o = (C - centre) * s
    dd = d * s
    a = np.sum(dd * dd, 1)
    b = dd @ o
    c = o @ o - 1.0
    disc = b * b - a * c
    return np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)


def render(cam, scale=2):
    """Half-resolution z-depth and full-resolution mask of the subject for one camera."""
    w, h = W // scale, H // scale
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * scale - 0.5, (np.arange(h) + 0.5) * scale - 0.5)
    d_cam = np.stack([(uu - W / 2) / FOCAL, (vv - H / 2) / FOCAL, np.ones_like(uu)], -1).reshape(-1, 3)
    d = d_cam @ cam["R"]  # world directions, unit z-depth parameter
    t = np.minimum(ray_ellipsoid(cam["C"], d, BODY_C, BODY_AX),
                   ray_ellipsoid(cam["C"], d, HEAD_C, np.full(3, HEAD_R)))
    depth = np.where(np.isfinite(t), t, 0.0).reshape(h, w)
    # full-resolution mask: the depth silhouette upsampled, slightly dilated
    m = cv2.resize((depth > 0).astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
    return depth, cv2.dilate(m, np.ones((3, 3), np.uint8))


def write_rig(root: Path, frames=(1, 2, 3), offsets=TRUE_OFFSETS):
    cams = make_cams()
    ref = {"cameras": []}
    for ci, c in enumerate(cams):
        ref["cameras"].append({"folder": c["folder"], "image_name": "000001.jpg", "camera_model_name": "PINHOLE",
                               "params": [FOCAL, FOCAL, W / 2, H / 2], "width": W, "height": H,
                               "R": c["R"].tolist(), "t": c["t"].tolist()})
        depth, mask = render(c)
        meas = np.where(depth > 0, depth - offsets[ci], 0.0)
        png = np.clip(np.round(meas * 1000), 0, 65535).astype(np.uint16)
        for f in frames:
            for sub, arr in (("depth_registered", png), ("masks_refined", mask)):
                p = root / sub / c["folder"] / f"{f:06d}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), arr)
    (root / "reference_export.json").write_text(json.dumps(ref))
    return cams


def run(root: Path, *extra):
    out = root / "depth_offsets.json"
    with contextlib.redirect_stdout(io.StringIO()):
        assert MODULE.main(["--data-dir", str(root), "--min-points", "100", *extra]) == 0
    return json.loads(out.read_text())


class SolveTests(unittest.TestCase):
    def test_mean_gauge_recovers_differences_from_exact_equations(self):
        rng = np.random.default_rng(1)
        o = rng.normal(0, 0.01, 5)
        eqs = []
        for i in range(5):
            for j in range(5):
                if i != j:
                    c = rng.uniform(0.3, 0.9)
                    eqs.append((i, j, -(c * o[i] - o[j]), c, 1.0))
        est, res, _ = MODULE.solve(eqs, 5, "mean", 0.0, 0)
        np.testing.assert_allclose(est, o - o.mean(), atol=1e-9)
        est, res, _ = MODULE.solve(eqs, 5, "free", 0.0, 0)
        np.testing.assert_allclose(est, o, atol=1e-9)  # c != 1 makes the common shift identifiable

    def test_irls_downweights_a_bad_pair(self):
        o = np.array([0.004, -0.002, 0.0, -0.002])
        eqs = [(i, j, -(0.5 * o[i] - o[j]), 0.5, 1.0) for i in range(4) for j in range(4) if i != j]
        eqs[0] = (eqs[0][0], eqs[0][1], eqs[0][2] + 0.03, 0.5, 1.0)  # 3 cm outlier
        est, _, w = MODULE.solve(eqs, 4, "mean", 0.002, 20)
        ls, _, _ = MODULE.solve(eqs, 4, "mean", 0.0, 0)
        err, err_ls = np.abs(est - o).max(), np.abs(ls - o).max()
        self.assertLess(err, 0.001)
        self.assertGreater(err_ls, 3 * err)
        self.assertLess(w[0], 0.2)


class SyntheticRigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "rig"
        cls.cams = write_rig(cls.root)
        # colour-triangulated stand-in for the gauge check: exact surface points
        pts = []
        for cam in cls.cams:
            depth, _ = render(cam)
            v, u = np.nonzero(depth > 0)
            z = depth[v, u]
            x = (((u + 0.5) * 2 - 0.5) - W / 2) / FOCAL * z
            y = (((v + 0.5) * 2 - 0.5) - H / 2) / FOCAL * z
            Xc = np.stack([x, y, z], 1)[::25]
            pts.append((Xc - cam["t"]) @ cam["R"])
        np.save(cls.root / "surface_points.npy", np.concatenate(pts).astype(np.float32))
        cls.out = run(cls.root, "--gauge-points", "surface_points.npy")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def offsets(self, out):
        return np.array([out["offsets_m"][c["folder"]] for c in self.cams])

    def test_contract_and_schema(self):
        out = self.out
        self.assertEqual(out["contract"], MODULE.CONTRACT)
        self.assertIn("ADD", out["contract"])
        self.assertEqual(out["gauge"], "mean")
        self.assertEqual(out["frames"], [1, 2, 3])
        self.assertEqual(sorted(out["offsets_m"]), [f"{i + 1:04d}" for i in range(N_CAMS)])
        for k in ("pair_residual_rms_mm", "per_pair", "per_frame_offsets_mm", "per_frame_spread_mm", "free_solution"):
            self.assertIn(k, out)
        self.assertTrue(all("->" in k for k in out["per_pair"]))

    def test_mean_gauge_recovers_offset_differences(self):
        est = self.offsets(self.out)
        self.assertAlmostEqual(float(est.mean()), 0.0, places=9)
        np.testing.assert_allclose(est, TRUE_OFFSETS - TRUE_OFFSETS.mean(), atol=0.0005)

    def test_pair_residuals_collapse(self):
        rms = self.out["pair_residual_rms_mm"]
        self.assertGreater(rms["before"], 5.0)
        self.assertLess(rms["after"], 0.5)
        self.assertLess(max(self.out["per_frame_spread_mm"].values()), 0.3)

    def test_neighbours_only_far_side_rejected(self):
        # 60 deg ring: opposite cameras see the other side of the subject and must not pair
        for k in self.out["per_pair"]:
            i, j = (int(x) - 1 for x in k.split("->"))
            self.assertIn((i - j) % N_CAMS, (1, N_CAMS - 1), k)

    def test_free_solution_keeps_differences_common_shift_weak(self):
        free = self.out["free_solution"]
        est = np.array([free["offsets_mm"][c["folder"]] for c in self.cams]) / 1000
        np.testing.assert_allclose(est - est.mean(), TRUE_OFFSETS - TRUE_OFFSETS.mean(), atol=0.0005)
        # surfaces seen by two cameras face both: the common shift is barely observable
        self.assertLess(free["conditioning"], 0.2)

    def test_gauge_points_report_sign_and_shift(self):
        gp = self.out["gauge_points"]
        self.assertIsNotNone(gp)
        per = np.array([gp["per_camera_mm"][c["folder"]]["offset_mm"] for c in self.cams]) / 1000
        np.testing.assert_allclose(per, TRUE_OFFSETS, atol=0.001)
        self.assertAlmostEqual(gp["common_shift_mm"] / 1000, TRUE_OFFSETS.mean(), delta=0.0005)

    def test_no_bias_gives_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "rig"
            write_rig(root, frames=(1, 2), offsets=np.zeros(N_CAMS))
            out = run(root, "--gauge-points", "none")
            free = run(root, "--gauge", "free", "--gauge-points", "none", "--huber-mm", "2")
        np.testing.assert_allclose(self.offsets(out), 0.0, atol=0.0005)
        self.assertIsNone(out["gauge_points"])
        self.assertEqual(free["gauge"], "free")
        self.assertTrue(np.isfinite(self.offsets(free)).all())


class HelperTests(unittest.TestCase):
    def test_pick_frames_spreads_over_clip(self):
        self.assertEqual(MODULE.pick_frames(set(range(1, 378)), 3), [1, 189, 377])
        self.assertEqual(MODULE.pick_frames({1, 2, 3}, 12), [1, 2, 3])

    def test_edge_safe_drops_step_keeps_plane(self):
        d = np.full((20, 20), 1.5, np.float32)
        d[:, 10:] = 1.0
        ok = MODULE.edge_safe(d, 2, 0.015, 0.0, 0)
        self.assertTrue(ok[5, 3] and ok[5, 16])
        self.assertFalse(ok[5, 8:12].any())

    def test_load_points_colmap_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "points3D.txt"
            p.write_text("# header\n1 0.5 1.0 -2.0 255 0 0 0.3 1 2 3 4\n2 1 2 3 0 0 0 0.1 1 1 2 2\n")
            np.testing.assert_allclose(MODULE.load_points(p), [[0.5, 1.0, -2.0], [1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
