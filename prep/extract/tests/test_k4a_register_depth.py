#!/usr/bin/env python3
"""Tests for the pixel-level steps of prep/extract/k4a_register_depth.py.

The full Kinect path (factory calibration, depth -> colour transform) is validated on real
captures; these tests pin down the cleaning, z-buffering and hole filling on synthetic
depth images.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "k4a_register_depth.py"
SPEC = importlib.util.spec_from_file_location("k4a_register_depth", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FlyingPixelMaskTest(unittest.TestCase):
    def test_step_edge_is_removed_and_flat_areas_kept(self):
        depth = np.full((20, 20), 1000, np.uint16)  # 1 m
        depth[:, 10:] = 2000                         # 2 m step at column 10
        depth[0, 0] = 0                              # invalid pixel
        keep = MODULE.flying_pixel_mask(depth, abs_tol_m=0.03, rel_tol=0.02)
        self.assertFalse(keep[5, 9])
        self.assertFalse(keep[5, 10])
        self.assertTrue(keep[5, 3])
        self.assertTrue(keep[5, 16])
        self.assertFalse(keep[0, 0])

    def test_invalid_neighbours_do_not_count_as_discontinuity(self):
        depth = np.full((9, 9), 1500, np.uint16)
        depth[4, 5] = 0
        keep = MODULE.flying_pixel_mask(depth, abs_tol_m=0.03, rel_tol=0.02)
        self.assertTrue(keep[4, 4])
        self.assertFalse(keep[4, 5])

    def test_smooth_slope_within_tolerance_is_kept(self):
        depth = (1000 + 5 * np.arange(16))[None, :].repeat(8, 0).astype(np.uint16)  # 5 mm per pixel
        keep = MODULE.flying_pixel_mask(depth, abs_tol_m=0.03, rel_tol=0.02)
        self.assertTrue(keep.all())


class ZBufferTest(unittest.TestCase):
    def test_nearest_point_wins(self):
        u = np.array([3, 3], np.int32)
        v = np.array([2, 2], np.int32)
        z = np.array([2.0, 1.0], np.float32)
        zb = MODULE.zbuffer(u, v, z, w=8, h=6, splat=1)
        self.assertAlmostEqual(float(zb[2, 3]), 1.0)
        self.assertEqual(int((zb > 0).sum()), 1)

    def test_splat_covers_square_and_clips_at_border(self):
        u = np.array([0, 7], np.int32)
        v = np.array([0, 5], np.int32)
        z = np.array([1.0, 1.5], np.float32)
        zb = MODULE.zbuffer(u, v, z, w=8, h=6, splat=2)
        self.assertTrue((zb[0:2, 0:2] == 1.0).all())
        self.assertEqual(float(zb[5, 7]), 1.5)   # the other three splat pixels are off-image
        self.assertEqual(int((zb > 0).sum()), 5)


class FillPinholesTest(unittest.TestCase):
    def test_hole_in_flat_surface_is_filled(self):
        zb = np.full((7, 7), 1.2, np.float32)
        zb[3, 3] = 0
        out = MODULE.fill_pinholes(zb, tol_m=0.03)
        self.assertAlmostEqual(float(out[3, 3]), 1.2, places=5)

    def test_hole_on_depth_edge_stays_invalid(self):
        zb = np.full((7, 7), 1.0, np.float32)
        zb[:, 4:] = 2.0
        zb[3, 3] = 0
        out = MODULE.fill_pinholes(zb, tol_m=0.03)
        self.assertEqual(float(out[3, 3]), 0.0)

    def test_sparse_holes_are_not_invented(self):
        zb = np.zeros((7, 7), np.float32)
        zb[3, 2] = zb[3, 4] = 1.0
        out = MODULE.fill_pinholes(zb, tol_m=0.03)
        self.assertEqual(float(out[3, 3]), 0.0)


if __name__ == "__main__":
    unittest.main()
