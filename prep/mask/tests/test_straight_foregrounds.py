#!/usr/bin/env python3
"""Tests for prep/mask/straight_foregrounds.py.

Scene: a red disc (the subject) on a green background. The mask is the disc dilated by
8 px with a feathered edge, so its outer ring covers green background, as a safety margin
does on real masks. The straight foreground must carry subject colour (red) over the whole
mask support and never the background colour.
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


SCRIPT = Path(__file__).resolve().parents[1] / "straight_foregrounds.py"
SPEC = importlib.util.spec_from_file_location("straight_foregrounds", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

H, W, R = 120, 100, 25
RED, GREEN = (30, 40, 200), (40, 180, 40)  # BGR


def scene():
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = GREEN
    cv2.circle(img, (W // 2, H // 2), R, RED, -1)
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (W // 2, H // 2), R + 8, 255, -1)
    soft = cv2.GaussianBlur(mask, (0, 0), 2.0)
    mask = np.where(mask > 0, np.maximum(soft, 1), 0).astype(np.uint8)
    mask[cv2.erode(mask, np.ones((9, 9), np.uint8)) == 255] = 255
    return img, mask


class StraightForegroundTest(unittest.TestCase):
    def test_core_keeps_captured_colour_and_outside_is_black(self):
        img, mask = scene()
        fg = MODULE.straight_foreground(img, mask, core_erode=10, inpaint_radius=5)
        core = cv2.erode((mask >= 250).astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
        np.testing.assert_array_equal(fg[core], img[core])
        self.assertTrue((fg[mask == 0] == 0).all())

    def test_ring_carries_subject_colour_not_background(self):
        img, mask = scene()
        fg = MODULE.straight_foreground(img, mask, core_erode=10, inpaint_radius=5)
        yy, xx = np.mgrid[:H, :W]
        r = np.hypot(yy - H // 2, xx - W // 2)
        ring = (mask > 0) & (r > R + 1)          # mask support over green background
        self.assertGreater(int(ring.sum()), 100)
        px = fg[ring].astype(np.int32)
        self.assertTrue((px[:, 2] > px[:, 1]).all(), "ring pixels must be red-dominant")
        self.assertLess(float(np.abs(px - np.array(RED)).mean()), 20.0)

    def test_without_core_the_image_is_only_cut_out(self):
        img, mask = scene()
        thin = np.where(mask > 0, 128, 0).astype(np.uint8)  # no pixel reaches the solid core
        fg = MODULE.straight_foreground(img, thin, core_erode=3, inpaint_radius=5)
        np.testing.assert_array_equal(fg[thin > 0], img[thin > 0])

    def test_cli_writes_outputs_and_reads_erode_from_matching_report(self):
        img, mask = scene()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for cam in ("0001", "0002"):
                (d / "images" / cam).mkdir(parents=True)
                (d / "masks_refined" / cam).mkdir(parents=True)
                for f in (1, 2):
                    cv2.imwrite(str(d / "images" / cam / f"{f:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    cv2.imwrite(str(d / "masks_refined" / cam / f"{f:06d}.png"), mask)
            (d / "mask_report").mkdir()
            (d / "mask_report" / "masks.json").write_text(json.dumps(
                {"output_dir": str(d / "masks"), "params": {"dilate_px": 6, "feather_px": 2.0}}))
            (d / "mask_report" / "refine.json").write_text(json.dumps(
                {"output_dir": str(d / "masks_refined"), "params": {"dilate_px": 1, "feather_px": 1.0}}))
            with contextlib.redirect_stdout(io.StringIO()):
                rc = MODULE.main(["--data-dir", str(d), "--masks-dir", "masks_refined",
                                  "--output-dir", "foregrounds_refined", "--workers", "1"])
            self.assertEqual(rc, 0)
            done = json.loads((d / "foregrounds_refined" / "_complete.json").read_text())
            self.assertEqual(done["core_erode"], 3)          # 1 + 2 * 1.0, from refine.json
            self.assertEqual(done["frames"], [1, 2])
            self.assertEqual(done["cameras"], ["0001", "0002"])
            out = cv2.imread(str(d / "foregrounds_refined" / "0002" / "000002.png"))
            self.assertEqual(out.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
