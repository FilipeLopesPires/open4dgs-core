#!/usr/bin/env python3
"""Straight (un-premultiplied) foreground RGB for masked composite training.

``trigs-train-colmap --foregrounds-root`` with ``--config.masked-rgb-composite`` mattes the
ground truth onto a constant plate as ``fg * mask + plate * (1 - mask)``.  The captured image
cannot be used as ``fg`` directly: the mask's safety margin and soft edge contain real
background (floor, walls), which the model would then learn as an opaque halo.

This script keeps the captured colour where the subject is certain (the mask's solid core,
eroded by ``--core-erode`` pixels to step past the safety dilation and feather) and fills the
rest of the mask's support by inpainting outward from that core, so the soft edge carries
subject colour only.  Pixels outside the mask are written black (they are never used).

Inputs (capture data directory): ``images/<cam>/<frame:06d>.jpg`` and
``masks/<cam>/<frame:06d>.png`` (any mask source).  Output:
``foregrounds/<cam>/<frame:06d>.png`` plus ``foregrounds/_complete.json`` (written last).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def straight_foreground(img: np.ndarray, mask: np.ndarray, core_erode: int, inpaint_radius: int) -> np.ndarray:
    support = mask > 0
    core = mask >= 250
    if core_erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * core_erode + 1, 2 * core_erode + 1))
        core = cv2.erode(core.astype(np.uint8), k).astype(bool)
    out = img.copy()
    out[~support] = 0
    ring = support & ~core
    if ring.any() and core.any():
        # inpaint only a crop around the ring (much faster than the full frame)
        ys, xs = np.nonzero(support)
        pad = 4 * inpaint_radius + 2
        y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)
        crop = img[y0:y1, x0:x1].copy()
        crop[~core[y0:y1, x0:x1]] = 0
        holes = (~core[y0:y1, x0:x1]).astype(np.uint8) * 255
        filled = cv2.inpaint(crop, holes, inpaint_radius, cv2.INPAINT_TELEA)
        r = ring[y0:y1, x0:x1]
        out[y0:y1, x0:x1][r] = filled[r]
    return out


_G: dict = {}


def _init(data: str, images: str, masks: str, out: str, ext: str, erode: int, radius: int):
    _G.update(data=Path(data), images=images, masks=masks, out=Path(out), ext=ext, erode=erode, radius=radius)


def _do(job: tuple[str, int]) -> str | None:
    cam, f = job
    d = _G["data"]
    img = cv2.imread(str(d / _G["images"] / cam / f"{f:06d}.{_G['ext']}"), cv2.IMREAD_COLOR)
    m = cv2.imread(str(d / _G["masks"] / cam / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or m is None:
        return f"{cam}/{f:06d}: missing image or mask"
    if m.shape != img.shape[:2]:
        return f"{cam}/{f:06d}: mask {m.shape} != image {img.shape[:2]}"
    fg = straight_foreground(img, m, _G["erode"], _G["radius"])
    cv2.imwrite(str(_G["out"] / cam / f"{f:06d}.png"), fg, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--images-dir", default="images")
    ap.add_argument("--masks-dir", default="masks")
    ap.add_argument("--output-dir", default="foregrounds")
    ap.add_argument("--image-ext", default="jpg")
    ap.add_argument("--frame-start", type=int, default=1)
    ap.add_argument("--frame-end", type=int, default=0, help="inclusive; 0 = every frame that has a mask")
    ap.add_argument("--core-erode", type=int, default=0,
                    help="pixels to erode the solid mask core; 0 = read dilate_px + 2*feather_px "
                         "from mask_report/masks.json (fallback 10)")
    ap.add_argument("--inpaint-radius", type=int, default=5)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    data = args.data_dir.resolve()
    cams = sorted(p.name for p in (data / args.masks_dir).iterdir() if p.is_dir())
    erode = args.core_erode
    if erode <= 0:
        # Step past the mask's safety dilation and feather, read from the report of the
        # tool that wrote this mask set (depth_masks.py -> masks.json, refine_masks.py ->
        # refine.json; both record their output_dir).
        erode = 10
        for rep in sorted((data / "mask_report").glob("*.json")):
            try:
                j = json.loads(rep.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if Path(str(j.get("output_dir", ""))).name != Path(args.masks_dir).name:
                continue
            p = j.get("params", {})
            if "dilate_px" in p:
                erode = int(round(p["dilate_px"] + 2 * p.get("feather_px", 0)))
                break
    frames_avail = sorted(int(p.stem) for p in (data / args.masks_dir / cams[0]).glob("*.png"))
    end = args.frame_end or frames_avail[-1]
    frames = [f for f in frames_avail if args.frame_start <= f <= end]
    out = data / args.output_dir
    (out / "_complete.json").unlink(missing_ok=True)
    for c in cams:
        (out / c).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    jobs = [(c, f) for f in frames for c in cams]
    errors = []
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(str(data), args.images_dir, args.masks_dir, str(out),
                                       args.image_ext, erode, args.inpaint_radius)) as ex:
        for i, err in enumerate(ex.map(_do, jobs, chunksize=4)):
            if err:
                errors.append(err)
            if (i + 1) % 300 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    if errors:
        print("\n".join(errors[:20]))
        raise SystemExit(f"{len(errors)} foreground(s) failed")
    (out / "_complete.json").write_text(json.dumps({
        "frames": [frames[0], frames[-1]], "cameras": cams, "core_erode": erode,
        "inpaint_radius": args.inpaint_radius, "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    print(f"wrote {len(jobs)} foregrounds to {out} (core erode {erode} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
