#!/usr/bin/env python3
"""
Write reduced-resolution copies of generate_masks.py outputs for training with
`trigs-train-colmap --data-factor N`.

Why: with --data-factor the trainer still decodes every full-size mask and
foreground PNG on the CPU each step and only then shrinks it. At 4K that
decode dominates the step time. Copies at exactly the size the parser expects
(width // N, height // N) skip the decode cost and the resize; the trainer
still applies each camera's undistortion when it loads them.

Resampling:
  masks        INTER_AREA (box average) of the matte.
  foregrounds  alpha-weighted INTER_AREA. The straight RGB is averaged with
               alpha as the weight, so soft-edge colour is not darkened by the
               zeroed colour of transparent neighbours. The trainer composites
               straight RGB with the mask, so a plain average would leave a
               dark fringe around the subject; with weighting, compositing the
               small copy equals shrinking the full-size composite.

Output: <data-dir>/masks_f2/<cam>/000001.png, <data-dir>/foregrounds_f2/... for
--factor 2. Existing outputs are skipped (writes are atomic), so an
interrupted run can simply be restarted.

Usage:
    python prep/mask/downscale_mask_outputs.py --data-dir . --factor 2 \
        --start-frame 1 --end-frame 263
    trigs-train-colmap ... --data-factor 2 \
        --masks-root masks_f2 --foregrounds-root foregrounds_f2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def target_size(height: int, width: int, factor: int) -> tuple[int, int]:
    """(width, height) the trainer's parser uses for this factor."""
    return max(width // factor, 1), max(height // factor, 1)


def downscale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    size = target_size(mask.shape[0], mask.shape[1], factor)
    out = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return _to_dtype(out, mask.dtype)


def downscale_foreground(
    foreground: np.ndarray, factor: int, weight: np.ndarray | None = None,
) -> np.ndarray:
    """Alpha-weighted shrink of a straight-colour foreground (BGR or BGRA).

    The weight is the image's own alpha channel, else `weight` (the matching
    mask), else uniform.
    """
    size = target_size(foreground.shape[0], foreground.shape[1], factor)
    has_alpha = foreground.ndim == 3 and foreground.shape[2] == 4
    rgb = foreground[..., :3].astype(np.float32)
    if has_alpha:
        weight = foreground[..., 3]
    if weight is None:
        out = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        return _to_dtype(out, foreground.dtype)
    a = weight.astype(np.float32) / float(np.iinfo(weight.dtype).max)
    if a.shape != rgb.shape[:2]:
        raise ValueError(f"weight shape {a.shape} does not match foreground {rgb.shape[:2]}")
    num = cv2.resize(rgb * a[..., None], size, interpolation=cv2.INTER_AREA)
    den = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    out_rgb = np.where(den[..., None] > 1e-6, num / np.maximum(den, 1e-6)[..., None], 0.0)
    out_rgb = _to_dtype(out_rgb, foreground.dtype)
    if not has_alpha:
        return out_rgb
    out_a = _to_dtype(den * float(np.iinfo(weight.dtype).max), foreground.dtype)
    return np.dstack((out_rgb, out_a))


def _is_grey_alpha(image: np.ndarray) -> bool:
    """OpenCV decodes a grey+alpha PNG with IMREAD_UNCHANGED as BGRA with B == G == R."""
    return (image.ndim == 3 and image.shape[2] == 4 and np.array_equal(image[..., 0], image[..., 1])
            and np.array_equal(image[..., 0], image[..., 2]))


def _to_dtype(x: np.ndarray, dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    return np.clip(np.rint(x), info.min, info.max).astype(dtype)


def write_png_atomic(path: Path, image: np.ndarray) -> None:
    """Write a PNG without leaving a partial final file after interruption."""
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    try:
        if image.ndim == 3 and image.shape[2] == 2:  # grey + confidence alpha (OpenCV cannot write it)
            from PIL import Image

            Image.fromarray(image, mode="LA").save(str(temporary), format="PNG")
        elif not cv2.imwrite(str(temporary), image):
            raise OSError(f"OpenCV failed to write: {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _process(task: tuple[str, str, str, int, str | None]) -> str:
    kind, src, dst, factor, mask_src = task
    cv2.setNumThreads(1)
    image = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Could not read {src}")
    if kind == "mask":
        if _is_grey_alpha(image):
            # grey + confidence alpha (consistent_masks.py --unknown): shrink both channels
            out = np.dstack((downscale_mask(image[..., 0], factor), downscale_mask(image[..., 3], factor)))
        elif image.ndim == 3:
            raise ValueError(f"Expected a single-channel mask: {src} has shape {image.shape}")
        else:
            out = downscale_mask(image, factor)
        result = "ok"
    else:
        weight = None
        if not (image.ndim == 3 and image.shape[2] == 4) and mask_src and Path(mask_src).exists():
            weight = cv2.imread(mask_src, cv2.IMREAD_UNCHANGED)
            if weight is not None and weight.ndim == 3:  # grey + alpha mask: the grey is the matte
                weight = weight[..., 0]
        out = downscale_foreground(image, factor, weight)
        has_weight = (image.ndim == 3 and image.shape[2] == 4) or weight is not None
        result = "ok" if has_weight else "unweighted"
    write_png_atomic(Path(dst), out)
    return result


def _frame_files(cam_dir: Path, start: int | None, end: int | None, pattern: str) -> list[Path]:
    if start is None or end is None:
        return sorted(p for p in cam_dir.glob("*.png") if not p.name.startswith("."))
    return [cam_dir / pattern.format(frame=f) for f in range(start, end + 1)]


def build_tasks(
    data_dir: Path, masks_dir: str, foregrounds_dir: str, factor: int,
    start: int | None, end: int | None, pattern: str, overwrite: bool,
) -> tuple[list[tuple[str, str, str, int, str | None]], int, list[Path]]:
    tasks, skipped, missing = [], 0, []
    for kind, name in (("mask", masks_dir), ("foreground", foregrounds_dir)):
        if not name:
            continue
        root = data_dir / name
        if not root.is_dir():
            raise FileNotFoundError(f"{kind} directory not found: {root}")
        out_root = data_dir / f"{name}_f{factor}"
        cams = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
        for cam in cams:
            (out_root / cam.name).mkdir(parents=True, exist_ok=True)
            for src in _frame_files(cam, start, end, pattern):
                if not src.exists():
                    missing.append(src)
                    continue
                dst = out_root / cam.name / src.name
                if dst.exists() and not overwrite:
                    skipped += 1
                    continue
                mask_src = str(data_dir / masks_dir / cam.name / src.name) if masks_dir else None
                tasks.append((kind, str(src), str(dst), factor, mask_src))
    return tasks, skipped, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--factor", type=int, default=2,
                    help="Must equal the trainer's --data-factor.")
    ap.add_argument("--masks-dir", default="masks", help="Empty string to skip masks.")
    ap.add_argument("--foregrounds-dir", default="foregrounds",
                    help="Empty string to skip foregrounds.")
    ap.add_argument("--start-frame", type=int, default=None,
                    help="With --end-frame, only these frames (else every PNG).")
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--pattern", default="{frame:06d}.png")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if args.factor < 2:
        ap.error("--factor must be >= 2")
    if (args.start_frame is None) != (args.end_frame is None):
        ap.error("--start-frame and --end-frame go together")

    tasks, skipped, missing = build_tasks(
        args.data_dir, args.masks_dir, args.foregrounds_dir, args.factor,
        args.start_frame, args.end_frame, args.pattern, args.overwrite,
    )
    if missing:
        print(f"ERROR: {len(missing)} requested frames are missing, e.g. {missing[0]}",
              file=sys.stderr)
        return 1
    print(f"{len(tasks):,} files to write, {skipped:,} already done, "
          f"factor {args.factor}, {args.workers} workers")

    counts: dict[str, int] = {}
    t0 = time.time()
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        results = pool.map(_process, tasks, chunksize=8) if pool else map(_process, tasks)
        for i, result in enumerate(results, 1):
            counts[result] = counts.get(result, 0) + 1
            if i % 1000 == 0 or i == len(tasks):
                print(f"  {i:,}/{len(tasks):,}  ({time.time() - t0:.0f}s)", flush=True)
    finally:
        if pool:
            pool.shutdown()
    if counts.get("unweighted"):
        print(f"WARNING: {counts['unweighted']:,} foregrounds had no alpha channel and no "
              "matching mask; they were shrunk without alpha weighting.")
    for name in (args.masks_dir, args.foregrounds_dir):
        if name:
            print(f"wrote {args.data_dir / f'{name}_f{args.factor}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
