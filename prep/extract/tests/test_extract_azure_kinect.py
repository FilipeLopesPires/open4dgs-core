"""Unit tests for the Azure Kinect importer (no recordings or ffmpeg required)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "align"))

import k4a_mkv  # noqa: E402
import k4a_sync  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXTRACT = _load("extract_azure_kinect", HERE / "extract_azure_kinect.py")
CALIB = _load("calibrate_azure_kinect", HERE.parent / "align" / "calibrate_azure_kinect.py")


def _cam(params):
    return {"Intrinsics": {"ModelType": "CALIBRATION_LensDistortionModelBrownConrady", "ModelParameters": params},
            "MetricRadius": 1.7}


class ModeIntrinsicsTests(unittest.TestCase):
    def test_color_1080p_crop_and_scale(self):
        p = [0.5, 0.5, 0.48, 0.64, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0.002, 0.001]
        intr = k4a_mkv.mode_intrinsics(_cam(p), k4a_mkv._COLOR_MODES["1080P"])
        s = 1920 / 4096
        self.assertEqual((intr.width, intr.height), (1920, 1080))
        self.assertAlmostEqual(intr.fx, 0.48 * 4096 * s)
        self.assertAlmostEqual(intr.fy, 0.64 * 3072 * s)
        self.assertAlmostEqual(intr.cx, 2048 * s - 0.5)
        self.assertAlmostEqual(intr.cy, (1536 - 384) * s - 0.5)
        # SDK stores p2 before p1; OpenCV order is k1,k2,p1,p2,k3..k6
        np.testing.assert_allclose(intr.dist, [0.1, 0.2, 0.001, 0.002, 0.3, 0.4, 0.5, 0.6])

    def test_depth_nfov_unbinned_offset(self):
        p = [0.5, 0.5, 0.49, 0.49] + [0.0] * 10
        intr = k4a_mkv.mode_intrinsics(_cam(p), k4a_mkv._DEPTH_MODES["NFOV_UNBINNED"])
        self.assertEqual((intr.width, intr.height), (640, 576))
        self.assertAlmostEqual(intr.cx, 512 - 192 - 0.5)
        self.assertAlmostEqual(intr.cy, 512 - 180 - 0.5)
        self.assertAlmostEqual(intr.fx, 0.49 * 1024)


class RotationTests(unittest.TestCase):
    def test_rotated_projection_matches_rotated_pixels(self):
        K = np.array([[900.0, 0, 950], [0, 905, 530], [0, 0, 1]])
        W, H = 1920, 1080
        X = np.array([0.3, -0.2, 2.5])
        u, v = (K @ (X / X[2]))[:2]
        img = np.zeros((H, W), np.uint8)
        for rot in ("n", "l", "r", "u"):
            K2, W2, H2 = EXTRACT.rotate_K(K, W, H, rot)
            Xr = EXTRACT.rotation_matrix_for_image(rot) @ X
            u2, v2 = (K2 @ (Xr / Xr[2]))[:2]
            # put a dot at (u,v), rotate the image, find the dot
            im = img.copy()
            im[int(round(v)), int(round(u))] = 255
            rim = EXTRACT.rotate_image(im, rot)
            self.assertEqual(rim.shape, (H2, W2))
            yy, xx = np.argwhere(rim == 255)[0]
            self.assertLess(abs(xx - u2), 1.0, rot)
            self.assertLess(abs(yy - v2), 1.0, rot)


class SyncTests(unittest.TestCase):
    def test_correlate_lag_recovers_shift(self):
        rng = np.random.default_rng(0)
        base = np.abs(np.convolve(rng.normal(size=600), np.ones(3) / 3, "same")) + 0.01
        a = base[10:410]
        b = base[7:407]  # b[i + 3] == a[i]
        cont, peak, _ = k4a_sync.correlate_lag(a, b, max_lag=10)
        self.assertAlmostEqual(cont, 3.0, delta=0.3)
        self.assertGreater(peak, 0.9)

    def test_subordinate_delay_resolution(self):
        # 18.14 ms at 30 fps = 0.544 frames; true n = 2 -> continuous lag 1.456
        lag, off, n, resid = k4a_sync.resolve_subordinate_lag(1.7, 0.5442)
        self.assertEqual(n, 2)
        self.assertEqual(lag, 1)
        self.assertAlmostEqual(off, 0.5442 - 1)
        self.assertLess(resid, 0.3)
        # small delay (daisy chain 160 us) keeps the integer lag
        lag, off, n, _ = k4a_sync.resolve_subordinate_lag(1.02, 0.0048)
        self.assertEqual((lag, n), (1, 1))
        self.assertAlmostEqual(off, 0.0048)


class CalibrationGeometryTests(unittest.TestCase):
    def test_look_at_point(self):
        target = np.array([0.2, 1.0, -0.3])
        centers = np.array([[2, 1.7, 0], [-2, 1.7, 0], [0, 1.8, 2], [0, 1.6, -2.0]])
        dirs = target - centers
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        np.testing.assert_allclose(CALIB.look_at_point(centers, dirs), target, atol=1e-9)

    def test_fit_floor(self):
        rng = np.random.default_rng(1)
        floor = np.c_[rng.uniform(-3, 3, 3000), np.zeros(3000), rng.uniform(-3, 3, 3000)]
        clutter = rng.uniform([-3, 0.1, -3], [3, 2.5, 3], (1500, 3))
        pts = np.r_[floor, clutter]
        cams = np.array([[1.5, 1.7, 0], [-1.5, 1.7, 0.0]])
        nrm, d, support = CALIB.fit_floor(pts, np.array([0.1, 1.0, 0.0]) / np.hypot(0.1, 1), cams)
        self.assertGreater(nrm[1], 0.999)
        self.assertAlmostEqual(d, 0.0, delta=0.01)
        self.assertGreaterEqual(support, 2900)

    def test_qvec_roundtrip(self):
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler("xyz", [20, -50, 130], degrees=True).as_matrix()
        q = CALIB.rotation_to_qvec(R)
        self.assertGreaterEqual(q[0], 0)
        R2 = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        np.testing.assert_allclose(R, R2, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
