#!/usr/bin/env python3
"""Tests for prep/mask/consistent_masks.py on a synthetic rig.

Scene: world y up, floor plane y = 0, a floating box (0.7 m wide, 0.4-1.1 m above the floor),
nothing else (rays that miss both have no depth, like ToF into the distance). Cameras on a
circle of 2.5 m, raised 2.4 m (about 33 degrees down) and looking at the box centre, so all
of them see its flat top; images 240x320, registered depth at half resolution. Camera 0's
mask misses the top rows of the box; the other cameras' masks are exact.
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
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "consistent_masks.py"
SPEC = importlib.util.spec_from_file_location("consistent_masks", SCRIPT)
CM = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CM
SPEC.loader.exec_module(CM)
RM = CM.rm

W, H = 240, 320
F = 300.0
BOX_MIN = np.array([-0.35, 0.4, -0.35])
BOX_MAX = np.array([0.35, 1.1, 0.35])
CENTRE = (BOX_MIN + BOX_MAX) / 2


def look_at(cam_pos, target):
    """World->camera R, t (OpenCV: x right, y down, z forward), world y up."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ cam_pos


def make_cams(n=3, azimuths=None):
    azimuths = azimuths if azimuths is not None else [2 * np.pi * k / n for k in range(n)]
    K = np.array([[F, 0, (W - 1) / 2], [0, F, (H - 1) / 2], [0, 0, 1.0]])
    cams = []
    for a in azimuths:
        pos = np.array([2.5 * np.sin(a), 2.4, 2.5 * np.cos(a)])
        R, t = look_at(pos, CENTRE)
        cams.append((K, R, t))
    return cams


def raycast(K, R, t, w, h):
    """(z-depth, hit-box) on a w x h grid covering the W x H image."""
    sx, sy = W / w, H / h
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * sx - 0.5, (np.arange(h) + 0.5) * sy - 0.5)
    dc = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)  # z = 1
    dw = dc @ R  # R^T d
    C = -R.T @ t
    with np.errstate(divide="ignore", invalid="ignore"):  # slab test
        t1 = (BOX_MIN - C) / dw
        t2 = (BOX_MAX - C) / dw
    tnear = np.nanmax(np.minimum(t1, t2), -1)
    tfar = np.nanmin(np.maximum(t1, t2), -1)
    ts = np.where((tnear <= tfar) & (tnear > 0), tnear, np.inf)
    tf = np.where(dw[..., 1] < -1e-6, -C[1] / np.where(dw[..., 1] < -1e-6, dw[..., 1], -1), np.inf)
    tf = np.where(tf > 0, tf, np.inf)
    tt = np.minimum(ts, tf)
    z = np.where(np.isfinite(tt), tt, 0.0).astype(np.float32)  # dc has z = 1, so t is z-depth
    return z, (ts < tf) & np.isfinite(ts)


def params(**kw):
    """The synthetic cameras have about 5x coarser depth pixels than the Kinect rig (2 cm at
    the box), so slanted faces need a looser edge-safe test."""
    return CM.ConsistentParams(**{"edge_tol": 0.1, **kw})


def geometry(K, R, t):
    return RM.DepthGeometry(K, np.zeros(4), W, H, R, t, up=np.array([0.0, 1.0, 0.0]), d=0.0, bias=0.0,
                            stage=(0.0, 0.0, 1.5), e1=np.array([1.0, 0.0, 0.0]), e2=np.array([0.0, 0.0, 1.0]))


def image_for(obj):
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), 90, np.uint8)
    img[obj] = (40, 150, 220)
    return np.clip(img.astype(np.int16) + rng.integers(-4, 5, img.shape), 0, 255).astype(np.uint8)


def scene(n=3, cut_rows=14, azimuths=None):
    """Returns (views, masks, imgs, gts, cut) with camera 0's mask missing the top
    ``cut_rows`` image rows of the box."""
    views, masks, imgs, gts = [], [], [], []
    cut = None
    for k, (K, R, t) in enumerate(make_cams(n, azimuths)):
        _, gt = raycast(K, R, t, W, H)
        z, _ = raycast(K, R, t, W // 2, H // 2)
        m = gt.copy()
        if k == 0:
            top = np.nonzero(gt.any(1))[0].min()
            cut = gt & (np.arange(H)[:, None] < top + cut_rows)
            m &= ~cut
        mask = (m * 255).astype(np.uint8)
        geo = geometry(K, R, t)
        views.append(CM.prepare_view(geo, mask, z))
        masks.append(mask)
        imgs.append(image_for(gt))
        gts.append(gt)
    return views, masks, imgs, gts, cut


class SupportTest(unittest.TestCase):
    def test_missing_cap_is_restored_from_two_views(self):
        views, masks, imgs, gts, cut = scene(3)
        out = CM.consistent_frame(views, masks, imgs, params())
        grey0 = out[0][0]
        restored = (grey0 >= 128) & cut
        self.assertGreater(restored.sum() / cut.sum(), 0.6, "most of the missing cap comes back")
        # nothing beyond the true silhouette (2 px tolerance for feather / guided filter)
        grown = (grey0 >= 128) & ~(cv2.dilate(gts[0].astype(np.uint8), np.ones((5, 5))) > 0)
        self.assertEqual(int(grown.sum()), 0)
        # the input mask is never reduced; exact masks of the other cameras barely change
        self.assertTrue(np.all(grey0 >= masks[0]))
        for k in (1, 2):
            self.assertTrue(np.all(out[k][0] >= masks[k]))
            extra = (out[k][0] >= 128) & ~gts[k]
            self.assertLess(extra.sum(), 0.02 * gts[k].sum())
        self.assertIsNone(out[0][1], "no unknown band unless asked for")

    def test_own_subject_depth_plus_one_view_restores(self):
        """keep_volume: camera 0 measures the box where its mask was cut, one other camera agrees."""
        views, masks, imgs, gts, cut = scene(2, azimuths=[0.0, 1.2])
        out = CM.consistent_frame(views, masks, imgs, params())
        self.assertGreater(((out[0][0] >= 128) & cut).sum() / cut.sum(), 0.6)
        out = CM.consistent_frame(views, masks, imgs, params(keep_volume=False))
        self.assertEqual(int(((out[0][0] >= 128) & cut).sum()), 0)

    def test_single_supporting_view_without_own_depth_gives_unknown(self):
        """Like dark hair: camera 0 has no depth where its mask was cut, one other camera sees it."""
        views, masks, imgs, gts, cut = scene(2, azimuths=[0.0, 1.2])
        v0 = views[0]
        cut_d = cv2.resize(cut.astype(np.uint8), (v0["z"].shape[1], v0["z"].shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
        hole = (cv2.dilate(cut_d.astype(np.uint8), np.ones((5, 5))) > 0) & (v0["mask_d"] < 0.5)
        v0["z"] = np.where(hole, 0, v0["z"]).astype(np.float32)
        v0["valid"] = v0["z"] > 0
        out = CM.consistent_frame(views, masks, imgs, params(unknown=True))
        grey0, conf0, info = out[0]
        self.assertEqual(int(((grey0 >= 128) & cut).sum()), 0, "one view is not enough to restore")
        self.assertIsNotNone(conf0)
        unknown = conf0 == 0
        self.assertGreater((unknown & cut).sum() / cut.sum(), 0.3)
        self.assertFalse(np.any(unknown & (grey0 >= 128)), "hard foreground stays known")
        self.assertGreater(info["unknown_frac"], 0)
        # with min_views 1 the same evidence restores
        out1 = CM.consistent_frame(views, masks, imgs, params(min_views=1))
        self.assertGreater(((out1[0][0] >= 128) & cut).sum() / cut.sum(), 0.5)

    def test_background_depth_blocks_restore(self):
        """A source point in front of camera j's measured background is only weak support."""
        views, masks, imgs, gts, cut = scene(3)
        # camera 0 now measures a wall 0.5 m behind the box centre where its mask was cut
        v0 = views[0]
        cut_d = cv2.resize(cut.astype(np.uint8), (v0["z"].shape[1], v0["z"].shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
        cut_d = cv2.dilate(cut_d.astype(np.uint8), np.ones((5, 5))) > 0
        zc = float(np.linalg.norm(make_cams(3)[0][1] @ CENTRE + make_cams(3)[0][2]))
        v0["z"] = np.where(cut_d & (v0["mask_d"] < 0.5), zc + 0.5, v0["z"]).astype(np.float32)
        v0["valid"] = v0["z"] > 0
        out = CM.consistent_frame(views, masks, imgs, params(unknown=True))
        self.assertLess(((out[0][0] >= 128) & cut).sum(), 0.1 * cut.sum())
        self.assertGreater(((out[0][1] == 0) & cut).sum(), 0.3 * cut.sum())


class NoDepthBlobTest(unittest.TestCase):
    def test_enclosed_no_depth_blob_next_to_mask(self):
        h, w = 200, 150  # the blob (300 px) must stay below nodepth_max_frac (2%) of the image
        m = np.zeros((h, w), bool)
        m[10:40, 15:45] = True  # subject
        valid = np.ones((h, w), bool)
        valid[40:55, 20:40] = False  # black chair shell below it, floor around
        bg = np.zeros((h, w), bool)
        hgt = np.full((h, w), 0.5, np.float32)
        hgt[55:] = 0.0
        p = CM.ConsistentParams()
        blob = CM.nodepth_blobs(m, valid, bg, hgt, p)
        self.assertEqual(int(blob.sum()), 15 * 20)
        # rimmed by background above floor level (a see-through gap): not a blob
        bg2 = bg.copy()
        bg2[38:57, 18:42] = True
        bg2 &= ~m
        bg2 &= valid
        self.assertEqual(int(CM.nodepth_blobs(m, valid, bg2, hgt, p).sum()), 0)
        # the same rim at floor level does not count as background
        hgt2 = np.zeros_like(hgt)
        self.assertEqual(int(CM.nodepth_blobs(m, valid, bg2, hgt2, p).sum()), 15 * 20)


class MaskFileTest(unittest.TestCase):
    def test_la_png_reads_as_plain_mask(self):
        grey = np.zeros((20, 30), np.uint8)
        grey[5:15, 5:25] = 255
        grey[4, 5:25] = 100
        conf = np.full_like(grey, 255)
        conf[0:4] = 0
        conf[4] = 128
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.png"
            CM.write_mask(p, grey, conf)
            self.assertEqual(Image.open(p).mode, "LA")
            np.testing.assert_array_equal(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), grey)
            np.testing.assert_array_equal(np.asarray(Image.open(p).convert("L")), grey)
            np.testing.assert_array_equal(cv2.imread(str(p), cv2.IMREAD_COLOR)[..., 1], grey)
            g, c = CM.read_mask(p)
            np.testing.assert_array_equal(g, grey)
            np.testing.assert_array_equal(c, conf)
            q = Path(tmp) / "plain.png"
            CM.write_mask(q, grey, None)
            self.assertEqual(Image.open(q).mode, "L")
            g, c = CM.read_mask(q)
            np.testing.assert_array_equal(g, grey)
            self.assertIsNone(c)

    def test_downscale_mask_outputs_keeps_grey_alpha(self):
        spec = importlib.util.spec_from_file_location(
            "downscale_mask_outputs", Path(__file__).resolve().parents[1] / "downscale_mask_outputs.py")
        ds = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ds)
        grey = np.zeros((20, 30), np.uint8)
        grey[4:16, 6:24] = 255
        conf = np.full_like(grey, 255)
        conf[:, :10] = 0
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "m.png", Path(tmp) / "m2.png"
            CM.write_mask(src, grey, conf)
            self.assertEqual(ds._process(("mask", str(src), str(dst), 2, None)), "ok")
            g, c = CM.read_mask(dst)
            self.assertEqual(g.shape, (10, 15))
            np.testing.assert_array_equal(g, cv2.resize(grey, (15, 10), interpolation=cv2.INTER_AREA))
            np.testing.assert_array_equal(c[:, :5], 0)
            np.testing.assert_array_equal(c[:, 5:], 255)
            # a foreground weighted by an LA mask uses its grey channel
            fg = np.full((20, 30, 3), 200, np.uint8)
            fsrc, fdst = Path(tmp) / "f.png", Path(tmp) / "f2.png"
            cv2.imwrite(str(fsrc), fg)
            self.assertEqual(ds._process(("foreground", str(fsrc), str(fdst), 2, str(src))), "ok")
            self.assertEqual(cv2.imread(str(fdst), cv2.IMREAD_UNCHANGED).shape, (10, 15, 3))


class CliTest(unittest.TestCase):
    def test_cli_writes_masks_report_and_sheet(self):
        cams = make_cams(3)
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            ref = {"cameras": []}
            for k, (K, R, t) in enumerate(cams):
                name = f"{k + 1:04d}"
                _, gt = raycast(K, R, t, W, H)
                z, _ = raycast(K, R, t, W // 2, H // 2)
                for f in (1, 2):
                    for sub in ("images", "masks_refined", "depth_registered"):
                        (data / sub / name).mkdir(parents=True, exist_ok=True)
                    m = gt.copy()
                    if k == 0:
                        top = np.nonzero(gt.any(1))[0].min()
                        m &= ~(np.arange(H)[:, None] < top + 14)
                    cv2.imwrite(str(data / "images" / name / f"{f:06d}.jpg"), image_for(gt))
                    cv2.imwrite(str(data / "masks_refined" / name / f"{f:06d}.png"), (m * 255).astype(np.uint8))
                    cv2.imwrite(str(data / "depth_registered" / name / f"{f:06d}.png"),
                                np.round(z * 1000).astype(np.uint16))
                ref["cameras"].append({"folder": name, "image_name": "000001.jpg", "camera_model_name": "PINHOLE",
                                       "params": [F, F, K[0, 2], K[1, 2]], "width": W, "height": H,
                                       "R": R.tolist(), "t": t.tolist()})
            (data / "reference_export.json").write_text(json.dumps(ref))
            (data / "mask_report").mkdir()
            (data / "mask_report" / "masks.json").write_text(json.dumps({
                "ground": {"has_floor": True, "up": [0, 1, 0], "d": 0.0, "e1": [1, 0, 0], "e2": [0, 0, 1]},
                "params": {"floor_bias": [0, 0, 0], "center": [0, 0], "radius": 1.5}}))
            (data / "mask_report" / "refine.json").write_text(json.dumps(
                {"output_dir": str(data / "masks_refined"), "params": {"dilate_px": 1, "feather_px": 1.0}}))
            with contextlib.redirect_stdout(io.StringIO()):
                rc = CM.main(["--data-dir", str(data), "--workers", "1", "--unknown", "--edge-tol", "0.1"])
            self.assertEqual(rc, 0)
            for k in range(3):
                for f in (1, 2):
                    self.assertTrue((data / "masks_consistent" / f"{k + 1:04d}" / f"{f:06d}.png").exists())
            rep = json.loads((data / "mask_report" / "consistent.json").read_text())
            self.assertEqual(Path(rep["output_dir"]).name, "masks_consistent")
            self.assertEqual(rep["params"]["dilate_px"], 1)  # read by straight_foregrounds.py
            self.assertTrue(rep["params"]["unknown"])
            self.assertGreater(rep["cameras"]["0001"]["area_gain_pct_median"], 1.0)
            self.assertTrue((data / "mask_report" / "consistent_sheet.jpg").exists())
            g = cv2.imread(str(data / "masks_consistent" / "0001" / "000001.png"), cv2.IMREAD_GRAYSCALE)
            m0 = cv2.imread(str(data / "masks_refined" / "0001" / "000001.png"), cv2.IMREAD_GRAYSCALE)
            self.assertTrue(np.all(g >= m0))
            with self.assertRaises(SystemExit):
                CM.main(["--data-dir", str(data), "--output-dir", str(data / "masks_refined")])


if __name__ == "__main__":
    unittest.main()
