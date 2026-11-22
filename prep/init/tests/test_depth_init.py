#!/usr/bin/env python3
"""Tests for prep/init/depth_init.py on a synthetic, analytically ray-cast depth rig.

Scene (+Y up, floor at y = 0, metres): six upright pinhole cameras on a ring (radius
1.5 m, height 1.2 m) looking at (0, 0.8, 0); a textured sphere (r = 0.3 m) centred at
(0, 0.8, 0) + V * (frame - 1), i.e. translating V per video frame; a grey floor.  Depth is
z-depth registered to each colour image (``depth_registered/<cam>/<frame>.npy``), so the
generic ``--depth-source registered`` path is exercised.  Masks mark the sphere.
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

SCRIPT = Path(__file__).resolve().parents[1] / "depth_init.py"
SPEC = importlib.util.spec_from_file_location("depth_init", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

W, H, FOCAL = 240, 320, 250.0
N_CAMS, RING_R, CAM_H = 6, 1.5, 1.2
TARGET = np.array([0.0, 0.8, 0.0])
C0, RADIUS = np.array([0.0, 0.8, 0.0]), 0.3
V = np.array([0.012, 0.0, -0.006])  # metres per video frame
STRAY_C, STRAY_R = np.array([0.95, 0.15, 0.2]), 0.04  # small static object inside the masks
FREQ = np.array([[21.0, 0.0, 13.0], [0.0, 17.0, 23.0], [19.0, 11.0, 0.0]])  # texture, rad/m per channel


def look_at(center, target):
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ center


def texture(local):
    return 0.5 + 0.4 * np.sin(local @ FREQ.T)


def make_cams(world_R=np.eye(3), world_t=np.zeros(3)):
    """Cameras; world_R/world_t re-express the same rig in another world frame (X' = R X + t)."""
    cams = []
    for i in range(N_CAMS):
        a = 2 * np.pi * i / N_CAMS + 0.3
        C = np.array([RING_R * np.cos(a), CAM_H, RING_R * np.sin(a)])
        R, t = look_at(C, TARGET)
        R2 = R @ world_R.T
        t2 = t - R2 @ world_t
        cams.append({"folder": f"{i + 1:04d}", "R": R2, "t": t2})
    return cams


def ray_sphere(C, d, centre, radius):
    oc = C - centre
    b = d @ oc
    a = np.sum(d * d, 1)
    disc = b * b - a * (oc @ oc - radius ** 2)
    return np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)


def render(cam, centre, world_R=np.eye(3), world_t=np.zeros(3), flying=False, stray=False):
    """z-depth, RGB (uint8) and mask of the scene for one camera. Scene geometry is defined in
    the canonical frame; the camera pose is in the (possibly transformed) world frame."""
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    d_cam = np.stack([(uu - W / 2) / FOCAL, (vv - H / 2) / FOCAL, np.ones_like(uu)], -1).reshape(-1, 3)
    # canonical-frame camera: R_c = R' world_R, C_c = world_R^T (C' - world_t)
    R = cam["R"] @ world_R
    C = world_R.T @ (-cam["R"].T @ cam["t"] - world_t)
    d = d_cam @ R  # world directions with unit z-depth parameter
    t_s = ray_sphere(C, d, centre, RADIUS)
    t_f = np.where(d[:, 1] < -1e-9, -C[1] / np.where(d[:, 1] < -1e-9, d[:, 1], -1), np.inf)
    t_o = ray_sphere(C, d, STRAY_C, STRAY_R) if stray else np.full_like(t_s, np.inf)
    t = np.minimum(np.minimum(t_s, t_f), t_o)
    sphere = np.isfinite(t_s) & (t_s <= t)
    other = np.isfinite(t_o) & (t_o <= t)
    depth = np.where(np.isfinite(t), t, 0.0).astype(np.float32)
    P = C + d * np.where(np.isfinite(t), t, 0)[:, None]
    rgb = np.full((len(P), 3), 0.45)
    rgb[sphere] = texture(P[sphere] - centre)
    rgb[other] = 0.95
    mask = sphere.reshape(H, W)
    in_mask = (sphere | other).reshape(H, W)
    depth = depth.reshape(H, W)
    if flying:  # mixed-depth pixels on the outer silhouette, as ToF sensors produce
        ring = (cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~mask
        fg_depth = cv2.dilate(np.where(mask, depth, 0).astype(np.float32), np.ones((3, 3), np.uint8))
        depth = np.where(ring, 0.5 * (depth + fg_depth), depth).astype(np.float32)
    rgb8 = np.clip(np.round(rgb.reshape(H, W, 3) * 255), 0, 255).astype(np.uint8)
    m8 = cv2.dilate(in_mask.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    return depth, rgb8, m8


def write_rig(root: Path, frames=range(1, 6), world_R=np.eye(3), world_t=np.zeros(3), sync=None,
              flying=False, stray=False):
    cams = make_cams(world_R, world_t)
    ref = {"cameras": []}
    for ci, c in enumerate(cams):
        ref["cameras"].append({"folder": c["folder"], "camera_model_name": "PINHOLE",
                               "params": [FOCAL, FOCAL, W / 2, H / 2], "width": W, "height": H,
                               "R": c["R"].tolist(), "t": c["t"].tolist()})
        for f in frames:
            # a camera with sync offset o captured frame f at time f + o
            o = 0.0 if sync is None else sync[c["folder"]]
            centre = C0 + V * (f - 1 + o)
            depth, rgb, mask = render(c, centre, world_R, world_t, flying=flying and ci == 0, stray=stray)
            for sub, arr, ext in (("images", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "png"), ("masks", mask, "png")):
                p = root / sub / c["folder"] / f"{f:06d}.{ext}"
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), arr)
            p = root / "depth_registered" / c["folder"] / f"{f:06d}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, depth)
    (root / "reference_export.json").write_text(json.dumps(ref))
    if sync is not None:
        (root / "sync.json").write_text(json.dumps({k: v / 30.0 for k, v in sync.items()}))


def run(root: Path, *extra, name="out.npz"):
    out = root / name
    args = ["--data-dir", str(root), "--depth-source", "registered", "--image-pattern", "{frame:06d}.png",
            "--frame-start", "1", "--frame-end", "5", "--keyframe-step", "2", "--target-points", "8000",
            "--output-path", str(out), "--workers", "2", *extra]
    with contextlib.redirect_stdout(io.StringIO()):
        assert MODULE.main(args) == 0
    with np.load(out) as z:
        return {k: z[k] for k in z.files}


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = np.sum(P * P, 1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x[:3], np.sqrt(x[3] + x[:3] @ x[:3])


class HelperTests(unittest.TestCase):
    def test_group_median(self):
        g = np.array([0, 0, 0, 1, 1, 2])
        v = np.array([[1.0], [5.0], [3.0], [2.0], [4.0], [9.0]])
        valid = np.array([True, True, True, True, True, False])
        med, cnt = MODULE.group_median(g, v, valid, 3)
        np.testing.assert_allclose(med[:, 0], [3.0, 3.0, 0.0])
        np.testing.assert_array_equal(cnt, [3, 2, 0])

    def test_choose_voxel_hits_target(self):
        P = np.random.default_rng(0).uniform(-0.5, 0.5, (50000, 3))
        vs = MODULE.choose_voxel(P, 5000)
        n = MODULE.voxel_groups(P, vs)[1]
        self.assertLessEqual(n, 5000)
        self.assertGreater(n, 4500)

    def test_edge_filter_drops_flying_pixel_keeps_plane(self):
        D = np.full((20, 20), 2.0, np.float32)
        D[:, 10:] = 1.0  # depth step
        D[5, 10] = 1.5  # flying pixel
        ok = MODULE.edge_ok(D, 1, 0.03, 0.01, 0)
        self.assertFalse(ok[5, 10])
        self.assertTrue(ok[5, 2] and ok[5, 17])
        self.assertFalse(ok[:, 9].any() or ok[:, 10].any())

    def test_zbuffer_keeps_nearest(self):
        cam = MODULE.dm.CameraPose("c", np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]]), np.zeros(4),
                                   100, 100, np.eye(3), np.zeros(3))
        wc = MODULE.WorkCam(cam, 1.0)
        pts = np.array([[0, 0, 2.0], [0, 0, 1.0], [0, 0, 3.0]], np.float32)
        D = MODULE.zbuffer_depth(wc, pts)
        self.assertAlmostEqual(float(D[50, 50]), 1.0)
        self.assertEqual(int((D > 0).sum()), 1)


class SyntheticRigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "rig"
        write_rig(cls.root, flying=True, stray=True)
        cls.z = run(cls.root, "--export-ply-dir", str(cls.root / "ply"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, frame):
        return self.z["source_frames"].reshape(-1) == frame

    def test_schema_matches_prep_roma(self):
        z = self.z
        n = len(z["positions"])
        spec = {"positions": (np.float32, (n, 3)), "colors": (np.float32, (n, 3)),
                "velocities": (np.float32, (n, 3)), "accelerations": (np.float32, (n, 3)),
                "has_acceleration": (np.bool_, (n,)), "track_flags": (np.uint8, (n,)),
                "times": (np.float32, (n, 1)), "durations": (np.float32, (n, 1)),
                "has_velocity": (np.bool_, (n,)), "is_static": (np.bool_, (n,)),
                "dynamic_protect": (np.bool_, (n,)), "dynamic_allowed": (np.bool_, (n,)),
                "stage_keep": (np.bool_, (n,)), "source_frames": (np.int32, (n, 1))}
        for k, (dt, shape) in spec.items():
            self.assertEqual(z[k].dtype, dt, k)
            self.assertEqual(z[k].shape, shape, k)
        np.testing.assert_array_equal(z["keyframe_frames"], [1, 3, 5])
        self.assertEqual(int(z["frame_start"]), 0)
        self.assertEqual(int(z["frame_end"]), 4)
        self.assertEqual(int(z["source_frame_start"]), 1)
        self.assertEqual(int(z["source_frame_end"]), 5)
        self.assertEqual(int(z["keyframe_step"]), 2)
        self.assertEqual(str(z["velocity_units"]), "world_per_video_frame")
        self.assertEqual(str(z["time_units"]), "normalized_0_1_over_source_frame_range")
        self.assertNotIn("kind", z)  # would switch the trainer's anchor handling on
        for f, t in ((1, 0.0), (3, 0.5), (5, 1.0)):
            np.testing.assert_allclose(z["times"][self.rows(f)], t)
        self.assertFalse(z["is_static"].any())
        self.assertTrue(z["dynamic_allowed"].all() and z["stage_keep"].all())
        self.assertTrue((self.root / "ply" / "depth_init_f000003.ply").exists())

    def test_points_on_sphere_no_floor_no_flying_pixels(self):
        for f in (1, 3, 5):
            P = self.z["positions"][self.rows(f)]
            self.assertLessEqual(len(P), 8000)
            self.assertGreater(len(P), 3000)
            err = np.abs(np.linalg.norm(P - (C0 + V * (f - 1)), axis=1) - RADIUS)
            self.assertLess(np.percentile(err, 99), 0.008, f)
            self.assertLess(err.max(), 0.02, f)

    def test_detached_masked_object_dropped(self):
        P = self.z["positions"]
        self.assertEqual(int((np.linalg.norm(P - STRAY_C, axis=1) < 0.1).sum()), 0)
        z = run(self.root, "--cluster-voxel", "0", name="out_nocluster.npz")
        self.assertGreater(int((np.linalg.norm(z["positions"] - STRAY_C, axis=1) < 0.1).sum()), 10)

    def test_colours_match_texture(self):
        f = 3
        P = self.z["positions"][self.rows(f)]
        centre = C0 + V * (f - 1)
        surf = centre + (P - centre) / np.linalg.norm(P - centre, axis=1, keepdims=True) * RADIUS
        err = np.abs(self.z["colors"][self.rows(f)] - texture(surf - centre))
        self.assertLess(np.median(err), 0.03)

    def test_velocity_recovers_translation(self):
        for f in (1, 3, 5):
            m = self.rows(f)
            v = self.z["velocities"][m][self.z["has_velocity"][m]]
            self.assertGreater(len(v), 0.9 * m.sum())
            med = np.median(v, axis=0)
            np.testing.assert_allclose(med, V, atol=0.0015, err_msg=f"frame {f}")
        m = self.rows(3)
        self.assertGreater(np.mean(self.z["track_flags"][m] == 3), 0.8)  # measured both ways (centred)
        self.assertGreater(self.z["has_acceleration"][m].mean(), 0.5)
        self.assertLess(np.median(np.linalg.norm(self.z["accelerations"][m], axis=1)), 0.003)

    def test_wider_velocity_baseline(self):
        z = run(self.root, "--velocity-delta", "2", name="out_d2.npz")
        m = z["source_frames"].reshape(-1) == 3  # flow partners 1 and 5
        np.testing.assert_allclose(np.median(z["velocities"][m], axis=0), V, atol=0.0015)
        self.assertLess(np.median(np.linalg.norm(z["accelerations"][m], axis=1)), 0.003)

    def test_merge_npz_appends_rows_at_keyframes(self):
        z = run(self.root, "--merge-npz", str(self.root / "out.npz"), name="merged.npz")
        n = len(self.z["positions"])
        self.assertEqual(len(z["positions"]), 2 * n)
        np.testing.assert_array_equal(z["source_frames"][n:], self.z["source_frames"])
        np.testing.assert_allclose(z["times"][n:], self.z["times"])
        np.testing.assert_allclose(z["durations"][n:], self.z["durations"], rtol=1e-6)
        self.assertEqual(str(z["mode"]), "depth_init_dynamic_merged")

    def test_trainer_loader_accepts_npz(self):
        try:
            import torch  # noqa: F401
            from trigs.init.from_colmap import build_canonical_from_npz
        except Exception as exc:  # pragma: no cover - trainer not installed
            self.skipTest(f"trigs not importable: {exc}")
        params, sh_degree, extent = build_canonical_from_npz(
            self.root / "out.npz", fps=30.0, keyframe_stride=2, device="cpu", compute_device="cpu",
            source_frame_start=1, source_frame_end=5, keep_static_when_filtering=True,
            filter_to_local_keyframe_stride=True, init_sigma_t_mode="npz")
        self.assertEqual(params["means"].shape[0], len(self.z["positions"]))
        self.assertGreater(extent, 0)


class SourceAgnosticTests(unittest.TestCase):
    def test_rigid_world_transform(self):
        """The same rig in a rotated, shifted world frame gives the transformed result."""
        ang = 0.7
        Rw = cv2.Rodrigues(np.array([0.2, ang, -0.1]))[0]
        tw = np.array([3.0, -1.0, 2.0])
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            write_rig(a)
            write_rig(b, world_R=Rw, world_t=tw)
            za, zb = run(a), run(b)
        for f in (1, 3, 5):
            ma = za["source_frames"].reshape(-1) == f
            mb = zb["source_frames"].reshape(-1) == f
            ca, ra = fit_sphere(za["positions"][ma])
            cb, rb = fit_sphere(zb["positions"][mb])
            np.testing.assert_allclose(Rw @ ca + tw, cb, atol=0.003)
            self.assertAlmostEqual(ra, rb, delta=0.003)
            np.testing.assert_allclose(Rw @ np.median(za["velocities"][ma], 0), np.median(zb["velocities"][mb], 0),
                                       atol=0.0015)

    def test_sync_offsets_compensated(self):
        """Cameras that captured half a frame late: points are moved back to the frame time."""
        sync = {f"{i + 1:04d}": 0.5 for i in range(N_CAMS)}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "rig"
            write_rig(root, sync=sync)
            on = run(root)
            off = run(root, "--sync-json", "none")
        m = on["source_frames"].reshape(-1) == 3
        c_on, _ = fit_sphere(on["positions"][m])
        c_off, _ = fit_sphere(off["positions"][off["source_frames"].reshape(-1) == 3])
        truth = C0 + V * 2
        self.assertLess(np.linalg.norm(c_on - truth), 0.002)
        self.assertGreater(np.linalg.norm(c_off - truth), 0.005)


class DepthOffsetTests(unittest.TestCase):
    def test_offset_source_moves_points_along_rays(self):
        class Fake:
            name = "fake"

            def points_cam(self, ci, frame):
                return np.array([[0.1, -0.2, 1.0], [0.3, 0.0, 2.0]], np.float32)

        src = MODULE.OffsetDepthSource(Fake(), np.array([0.0, 0.015]))
        P0 = Fake().points_cam(0, 1)
        np.testing.assert_array_equal(src.points_cam(0, 1), P0)  # zero offset: unchanged
        P = src.points_cam(1, 1)
        np.testing.assert_allclose(P[:, 2], P0[:, 2] + 0.015, atol=1e-6)
        np.testing.assert_allclose(P[:, :2] / P[:, 2:], P0[:, :2] / P0[:, 2:], atol=1e-6)
        self.assertEqual(src.name, "fake+offsets")

    def test_offsets_json_must_cover_every_camera(self):
        cams = [MODULE.dm.CameraPose(f, np.eye(3), np.zeros(4), 10, 10, np.eye(3), np.zeros(3)) for f in ("0001", "0002")]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "o.json"
            p.write_text(json.dumps({"contract": "...", "offsets_m": {"0001": 0.01, "0002": -0.002}}))
            np.testing.assert_allclose(MODULE.load_depth_offsets(p, cams), [0.01, -0.002])
            p.write_text(json.dumps({"offsets_m": {"0001": 0.01}}))
            with self.assertRaises(SystemExit):
                MODULE.load_depth_offsets(p, cams)

    def test_biased_depth_corrected(self):
        """Each camera measures depth too near/far by 2 cm: the offsets file restores the sphere."""
        offs = {f"{i + 1:04d}": (0.02 if i % 2 else -0.02) for i in range(N_CAMS)}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "rig"
            write_rig(root)
            for cam, o in offs.items():
                for p in (root / "depth_registered" / cam).glob("*.npy"):
                    D = np.load(p)
                    np.save(p, np.where(D > 0, D - o, 0).astype(np.float32))  # measured = true - offset
            (root / "depth_offsets.json").write_text(json.dumps({"offsets_m": offs}))
            raw = run(root, name="raw.npz")
            fixed = run(root, "--depth-offsets-json", "depth_offsets.json", name="fixed.npz")

        def surface_err(z, f):
            P = z["positions"][z["source_frames"].reshape(-1) == f]
            return np.abs(np.linalg.norm(P - (C0 + V * (f - 1)), axis=1) - RADIUS)

        for f in (1, 3, 5):
            self.assertLess(np.median(surface_err(fixed, f)), 0.003, f)
            self.assertGreater(np.median(surface_err(raw, f)), 2 * np.median(surface_err(fixed, f)), f)
        np.testing.assert_allclose(fixed["depth_offsets_m"], [offs[k] for k in sorted(offs)], atol=1e-7)
        self.assertEqual(list(fixed["depth_offsets_cameras"]), sorted(offs))
        self.assertNotIn("depth_offsets_m", raw)


if __name__ == "__main__":
    unittest.main()
