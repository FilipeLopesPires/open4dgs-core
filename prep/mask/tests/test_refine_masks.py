#!/usr/bin/env python3
"""Tests for prep/mask/refine_masks.py on synthetic images, coarse masks and depth.

Scene (portrait 240x400): light-grey "floor" in the lower part, a darker "wall" above; a
subject made of a dark torso (ellipse), light trousers (rectangle) and dark shoes. The
coarse mask is the subject dilated by 12 px (like a loose depth mask) plus a background
patch next to the head.
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


SCRIPT = Path(__file__).resolve().parents[1] / "refine_masks.py"
SPEC = importlib.util.spec_from_file_location("refine_masks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

W, H = 240, 400
WALL, FLOOR = (150, 160, 170), (200, 205, 210)
TORSO, TROUSERS, SHOE = (60, 50, 45), (190, 185, 185), (70, 40, 30)


def params(**kw):
    """Defaults scaled to the small synthetic images."""
    base = dict(core_erode_px=9, roi_pad_px=12, local_radius_px=12, person_grow_px=8, guided_radius=2)
    base.update(kw)
    return MODULE.RefineParams(**base)


def scene(seed=0, shift=0, patch_colour=WALL):
    """Returns (image BGR, ground-truth subject mask bool, coarse mask uint8, patch mask bool)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    img[:200] = WALL
    img[200:] = FLOOR
    gt = np.zeros((H, W), np.uint8)
    cx = 120 + shift
    cv2.ellipse(gt, (cx, 70), (22, 26), 0, 0, 360, 1, -1)  # head
    cv2.ellipse(gt, (cx, 160), (45, 70), 0, 0, 360, 1, -1)  # torso
    cv2.rectangle(gt, (cx - 35, 210), (cx + 35, 330), 1, -1)  # trousers
    cv2.rectangle(gt, (cx - 40, 330), (cx + 40, 350), 1, -1)  # shoes
    body = np.zeros_like(gt)
    cv2.ellipse(body, (cx, 70), (22, 26), 0, 0, 360, 1, -1)
    cv2.ellipse(body, (cx, 160), (45, 70), 0, 0, 360, 1, -1)
    img[body > 0] = TORSO
    trousers = np.zeros_like(gt)
    cv2.rectangle(trousers, (cx - 35, 210), (cx + 35, 330), 1, -1)
    img[(trousers > 0) & (body == 0)] = TROUSERS
    shoes = np.zeros_like(gt)
    cv2.rectangle(shoes, (cx - 40, 330), (cx + 40, 350), 1, -1)
    img[shoes > 0] = SHOE
    img = np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    patch = np.zeros((H, W), np.uint8)
    cv2.rectangle(patch, (cx + 22, 40), (cx + 45, 80), 1, -1)
    patch &= 1 - gt
    img[patch > 0] = patch_colour
    coarse = cv2.dilate(gt, MODULE._disk(12)) | patch
    return img, gt > 0, (coarse * 255).astype(np.uint8), patch > 0


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / (a | b).sum()


def refine(img, coarse, depth=None, geo=None, person=None, p=None):
    p = p or params()
    fg, info = MODULE.segment_frame(img, coarse, depth, geo, p, person)
    return MODULE.finish_mask(fg.astype(np.float32), img, coarse, info.get("sure_fg"), p, info.get("allowed")), info


class RefineMasksTests(unittest.TestCase):
    def test_colour_only_tightens_a_loose_mask(self):
        img, gt, coarse, _ = scene()
        out, _ = refine(img, coarse)
        self.assertEqual(out.shape, coarse.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(iou(out > 127, gt), 0.93)
        self.assertGreater(iou(out > 127, gt), iou(coarse > 127, gt) + 0.15)
        # Shoes (dark, like the torso) survive.
        self.assertGreater((out[332:348] > 127).sum() / gt[332:348].sum(), 0.9)

    def test_never_extends_beyond_the_dilated_coarse_mask(self):
        img, _, coarse, _ = scene()
        p = params()
        out, _ = refine(img, coarse, p=p)
        allowed = cv2.dilate((coarse > 127).astype(np.uint8), MODULE._disk(int(2 * p.max_grow_px) + 1)) > 0
        self.assertFalse(np.any((out > 0) & ~allowed))

    def test_soft_edge_is_narrow(self):
        img, _, coarse, _ = scene()
        out, _ = refine(img, coarse)
        soft = (out > 10) & (out < 245)
        hard = (out >= 128).astype(np.uint8)
        edge = cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        band = cv2.dilate(edge.astype(np.uint8), MODULE._disk(4)) > 0
        self.assertTrue(np.all(band[soft]))  # soft values only within 4 px of the edge

    def test_depth_removes_a_patch_with_subject_colour_behind_the_subject(self):
        # The patch has the torso colour, so colour cannot separate it; depth can.
        img, gt, coarse, patch = scene(patch_colour=TORSO)
        depth = np.full((H // 2, W // 2), 4.0, np.float32)  # wall 4 m
        depth[cv2.resize(gt.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) > 0] = 2.0
        no_depth, _ = refine(img, coarse)
        with_depth, info = refine(img, coarse, depth=depth)
        self.assertTrue(info["depth"])
        self.assertGreater((no_depth > 127)[patch].mean(), 0.5)  # colour alone keeps it
        self.assertLess((with_depth > 127)[patch].mean(), 0.05)
        self.assertGreater(iou(with_depth > 127, gt), 0.93)

    def test_depth_evidence_floor_is_a_candidate_not_background(self):
        z = np.full((40, 40), 2.0, np.float32)
        valid = np.ones_like(z, bool)
        hgt = np.full_like(z, 0.8)
        hgt[30:] = 0.01  # floor-level band (shoe toes look the same)
        z[:, 35:] = 3.0  # 1 m behind the subject
        coarse = np.ones_like(valid)
        core = np.zeros_like(valid)
        core[5:25, 5:30] = True
        bg, floor, fg = MODULE.depth_evidence(z, valid, hgt, None, coarse, core, params(local_radius_px=8))
        self.assertTrue(floor[32, 10] and not bg[32, 10] and not fg[32, 10])
        self.assertTrue(bg[10, 37] and not fg[10, 37])
        self.assertTrue(fg[10, 10])

    def test_person_hint_restores_clipped_toes_and_removes_unclaimed_patch(self):
        img, gt, coarse, patch = scene(patch_colour=TORSO)
        clipped = coarse.copy()
        clipped[338:] = 0  # coarse mask cut at the floor (like a depth floor margin)
        depth = np.full((H // 2, W // 2), 0.0, np.float32)  # no valid depth around the subject
        gts = cv2.resize(gt.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) > 0
        depth[gts] = 2.0
        person = (gt * 255).astype(np.uint8)
        p = params()
        out, info = refine(img, clipped, depth=depth, person=person, p=p)
        self.assertGreater((out[340:350] > 127)[gt[340:350]].mean(), 0.9)  # toes back
        # The patch has no valid depth and the torso colour; beyond the gap allowed around the
        # person mask (unclaimed_gap_px, work resolution = half size) it is removed.
        gap = 2 * p.unclaimed_gap_px + 2
        far = patch & ~(cv2.dilate(gt.astype(np.uint8), MODULE._disk(gap)) > 0)
        self.assertGreater(far.sum(), 300)
        self.assertLess((out > 127)[far].mean(), 0.02)
        self.assertLess((out > 127)[patch].mean(), 0.25)
        self.assertGreater(info["grown_frac_of_coarse"], 0)

    def test_temporal_vote_removes_single_frame_flicker(self):
        a = np.zeros((10, 10), np.uint8)
        a[2:8, 2:8] = 1
        b = a.copy()
        b[0, 0] = 1  # flicker in the centre frame only
        b[5, 5] = 0
        v = MODULE.temporal_vote([a, b, a])
        self.assertAlmostEqual(float(v[0, 0]), 1 / 3, places=6)  # spurious foreground removed
        self.assertAlmostEqual(float(v[5, 5]), 2 / 3, places=6)  # spurious hole filled
        np.testing.assert_array_equal(v > 0.5, a > 0)

    def test_guided_filter_keeps_a_constant_signal(self):
        g = np.random.default_rng(0).random((30, 40)).astype(np.float32)
        out = MODULE.guided_filter(g, np.full_like(g, 0.7), 3, 1e-3)
        np.testing.assert_allclose(out, 0.7, atol=1e-4)

    def test_parse_frames(self):
        self.assertEqual(MODULE.parse_frames("1-3,7, 5,3"), [1, 2, 3, 5, 7])

    def test_cli_writes_masks_report_and_sheet(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            for cam in ("0001", "0002"):
                (data / "images" / cam).mkdir(parents=True)
                (data / "masks" / cam).mkdir(parents=True)
                for f in (1, 2, 3):
                    img, _, coarse, _ = scene(seed=f, shift=f)
                    cv2.imwrite(str(data / "images" / cam / f"{f:06d}.jpg"), img)
                    cv2.imwrite(str(data / "masks" / cam / f"{f:06d}.png"), coarse)
            argv = ["--data-dir", str(data), "--sam", "off", "--workers", "1", "--core-erode-px", "9",
                    "--roi-pad-px", "12", "--local-radius-px", "12", "--guided-radius", "2"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(MODULE.main(argv), 0)
            for cam in ("0001", "0002"):
                for f in (1, 2, 3):
                    m = cv2.imread(str(data / "masks_refined" / cam / f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)
                    self.assertEqual(m.shape, (H, W))
                    self.assertEqual(m.dtype, np.uint8)
                    _, gt, _, _ = scene(seed=f, shift=f)
                    self.assertGreater(iou(m > 127, gt), 0.9)
            rep = json.loads((data / "mask_report" / "refine.json").read_text(encoding="utf-8"))
            self.assertEqual(set(rep["cameras"]), {"0001", "0002"})
            self.assertLess(rep["cameras"]["0001"]["area_ratio_median"], 0.9)
            self.assertTrue((data / "mask_report" / "refine_sheet.jpg").exists())
            # The coarse masks are never overwritten.
            with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
                MODULE.main(argv + ["--output-dir", str(data / "masks")])


if __name__ == "__main__":
    unittest.main()
