#!/usr/bin/env python3
"""Azure Kinect raw depth -> colour-registered depth maps (the generic depth contract).

Reads the raw depth written by ``extract_azure_kinect.py extract`` (``depth/<cam>/<frame>.png``,
uint16 mm in the depth camera's view) plus the factory depth intrinsics and depth->colour
transform from ``k4a_cameras.json``, and writes::

    depth_registered/<cam>/<frame:06d>.png   uint16 millimetres, 0 = invalid
    depth_registered/_complete.json          parameters + statistics (written last)

Each map is z-depth along the optical axis of that camera's undistorted (and rotated)
colour image, at ``1/--downscale`` of the image resolution.  This is the input contract of
``prep/mask/depth_masks.py --depth-source registered``, ``prep/init/depth_init.py`` and the
trainer's ``--depth-root``; other depth sensors only need their own converter to this form.

Method: depth pixels are cleaned of ToF "flying pixels" (local depth range test on the raw
image), lifted to 3D, moved into the colour camera, and z-buffered with a 2x2 splat (the
depth grid is about as dense as a half-resolution colour image, so single-pixel splats leave
pinholes).  Remaining one-pixel holes surrounded by consistent depth are filled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mask"))
import depth_masks as D  # noqa: E402


def flying_pixel_mask(depth_mm: np.ndarray, abs_tol_m: float, rel_tol: float) -> np.ndarray:
    """True where a raw depth pixel is valid and not on a depth discontinuity."""
    d = depth_mm.astype(np.float32) / 1000.0
    valid = d > 0
    big = np.where(valid, d, np.float32(1e4))
    small = np.where(valid, d, np.float32(0.0))
    k = np.ones((3, 3), np.uint8)
    lo = cv2.erode(big, k)
    hi = cv2.dilate(small, k)
    # neighbours that are invalid do not count as discontinuities
    lo = np.where(lo >= 1e4, d, lo)
    rng = np.maximum(hi - d, d - lo)
    return valid & (rng <= np.maximum(abs_tol_m, rel_tol * d))


def zbuffer(u: np.ndarray, v: np.ndarray, z: np.ndarray, w: int, h: int, splat: int) -> np.ndarray:
    zb = np.full((h, w), np.inf, np.float32)
    for du in range(splat):
        for dv in range(splat):
            uu, vv = u + du, v + dv
            inb = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            np.minimum.at(zb, (vv[inb], uu[inb]), z[inb])
    zb[~np.isfinite(zb)] = 0
    return zb


def fill_pinholes(zb: np.ndarray, tol_m: float) -> np.ndarray:
    """Fill invalid pixels with >= 6 valid 8-neighbours whose depths agree within tol."""
    valid = zb > 0
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    n = cv2.filter2D(valid.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    s = cv2.filter2D(zb, -1, k, borderType=cv2.BORDER_CONSTANT)
    hi = cv2.dilate(zb, np.ones((3, 3), np.uint8))
    lo = cv2.erode(np.where(valid, zb, np.float32(1e4)), np.ones((3, 3), np.uint8))
    cand = (~valid) & (n >= 6) & (hi - lo <= tol_m)
    out = zb.copy()
    out[cand] = s[cand] / n[cand]
    return out


_G: dict = {}


def _init(data_dir: str, downscale: int, out: str, abs_tol: float, rel_tol: float, splat: int):
    cams = D.load_cameras(Path(data_dir))
    _G.update(cams=cams, src=D.K4ARawDepthSource(Path(data_dir), cams), ds=downscale, out=Path(out),
              abs_tol=abs_tol, rel_tol=rel_tol, splat=splat)


def _do(job: tuple[int, int]) -> tuple[int, int, float]:
    ci, frame = job
    cams, src, ds = _G["cams"], _G["src"], _G["ds"]
    cam = cams[ci]
    rays, R, t = src.calib[ci]
    raw = cv2.imread(str(src.root / cam.folder / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"missing depth {cam.folder}/{frame:06d}.png")
    good = flying_pixel_mask(raw, _G["abs_tol"], _G["rel_tol"])
    z = raw[good].astype(np.float32) / 1000.0
    Xc = np.concatenate([rays[good] * z[:, None], z[:, None]], axis=1) @ R.T + t
    zc = Xc[:, 2]
    ok = zc > 0.1
    Xc, zc = Xc[ok], zc[ok]
    w, h = cam.width // ds, cam.height // ds
    if np.any(cam.dist != 0):
        uv = cv2.projectPoints(Xc.astype(np.float64), np.zeros(3), np.zeros(3), cam.K, cam.dist)[0][:, 0]
    else:
        uv = np.c_[cam.K[0, 0] * Xc[:, 0] / zc + cam.K[0, 2], cam.K[1, 1] * Xc[:, 1] / zc + cam.K[1, 2]]
    # full-res pixel centre convention -> downscaled grid; splat covers [floor, floor+splat)
    u = np.floor((uv[:, 0] + 0.5) / ds - 0.5 * (_G["splat"] - 1)).astype(np.int32)
    v = np.floor((uv[:, 1] + 0.5) / ds - 0.5 * (_G["splat"] - 1)).astype(np.int32)
    zb = fill_pinholes(zbuffer(u, v, zc.astype(np.float32), w, h, _G["splat"]), 0.03)
    dst = _G["out"] / cam.folder / f"{frame:06d}.png"
    cv2.imwrite(str(dst), np.clip(np.round(zb * 1000.0), 0, 65535).astype(np.uint16))
    return ci, frame, float((zb > 0).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output-dir", default="depth_registered")
    ap.add_argument("--downscale", type=int, default=2, help="output = image size / N (default 2)")
    ap.add_argument("--frame-start", type=int, default=1)
    ap.add_argument("--frame-end", type=int, default=0, help="inclusive; 0 = last extracted frame")
    ap.add_argument("--flying-abs", type=float, default=0.03, help="discontinuity tolerance, metres")
    ap.add_argument("--flying-rel", type=float, default=0.02, help="discontinuity tolerance, fraction of depth")
    ap.add_argument("--splat", type=int, default=2, help="splat size in output pixels")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    data = args.data_dir.resolve()
    cams = D.load_cameras(data)
    meta = json.loads((data / "k4a_cameras.json").read_text(encoding="utf-8"))
    end = args.frame_end or int(meta["num_frames"])
    frames = list(range(args.frame_start, end + 1))
    out = data / args.output_dir
    (out / "_complete.json").unlink(missing_ok=True)
    for c in cams:
        (out / c.folder).mkdir(parents=True, exist_ok=True)

    jobs = [(ci, f) for f in frames for ci in range(len(cams))]
    t0 = time.time()
    cover: dict[str, list[float]] = {c.folder: [] for c in cams}
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(str(data), args.downscale, str(out), args.flying_abs,
                                       args.flying_rel, args.splat)) as ex:
        for i, (ci, f, frac) in enumerate(ex.map(_do, jobs, chunksize=8)):
            cover[cams[ci].folder].append(frac)
            if (i + 1) % 300 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} maps ({time.time() - t0:.0f} s)", flush=True)

    report = {
        "contract": "uint16 millimetres, 0 = invalid, z-depth of the colour camera, registered to images/",
        "depth_scale": 0.001,
        "downscale": args.downscale,
        "resolution": {c.folder: [c.width // args.downscale, c.height // args.downscale] for c in cams},
        "frames": [frames[0], frames[-1]],
        "params": {"flying_abs": args.flying_abs, "flying_rel": args.flying_rel, "splat": args.splat},
        "valid_fraction_median": {k: float(np.median(v)) for k, v in cover.items()},
        "seconds": round(time.time() - t0, 1),
    }
    (out / "_complete.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {len(jobs)} maps to {out} in {report['seconds']} s; valid fraction (median) "
          + ", ".join(f"{k} {v:.2f}" for k, v in report["valid_fraction_median"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
