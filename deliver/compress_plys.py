#!/usr/bin/env python3
"""Convert per-frame Gaussian PLYs to the PlayCanvas compressed PLY format.

``frame_000000.ply`` becomes ``frame_000000.compressed.ply`` (gsply's compressed
writer: positions, rotations, scales and colours packed per 256-splat chunk into
32-bit words, higher SH bands as 8 bits). The files are about 3.9x smaller. SuperSplat
and the PlayCanvas engine load them; tools that only read the standard 3DGS PLY
layout (``x, y, z, f_dc_*, f_rest_*, opacity, scale_*, rot_*`` as floats) do not,
so this is an opt-in step (run_kinect_pipeline.ps1 -CompressPly).

Measured on frame 44 of the 3 s ``professor`` run (371 k splats): 87.7 MB -> 22.8 MB,
45.6 dB PSNR on the subject against the uncompressed PLY, rendered from the six
training cameras.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gsply

SUFFIX = ".compressed.ply"


def compressed_name(path: Path) -> Path:
    return path.with_name(path.name[: -len(".ply")] + SUFFIX)


def source_plys(folder: Path) -> list[Path]:
    """Uncompressed ``*.ply`` files in ``folder`` (not ``*.compressed.ply``), sorted."""
    return sorted(p for p in folder.glob("*.ply") if not p.name.endswith(SUFFIX))


def compress_folder(folder: Path, *, delete_source: bool = False) -> tuple[int, int, int]:
    """Compress every PLY in ``folder``; returns (files, bytes in, bytes out)."""
    files = source_plys(folder)
    if not files:
        raise SystemExit(f"no uncompressed .ply files in {folder}")
    n_in = n_out = 0
    for src in files:
        dst = compressed_name(src)
        # gsply appends ".compressed.ply" to any other name, so no temp file; the
        # source is only deleted after the write returned.
        gsply.plyread(src).save(dst, compressed=True)
        n_in += src.stat().st_size
        n_out += dst.stat().st_size
        if delete_source:
            src.unlink()
    return len(files), n_in, n_out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input-dir", required=True, type=Path, help="Folder of per-frame .ply files")
    ap.add_argument("--delete-source", action="store_true",
                    help="Delete each uncompressed .ply once its compressed copy is written")
    args = ap.parse_args(argv)
    if not args.input_dir.is_dir():
        raise SystemExit(f"{args.input_dir} is not a folder")
    n, n_in, n_out = compress_folder(args.input_dir, delete_source=args.delete_source)
    print(f"Compressed {n} PLYs in {args.input_dir}: {n_in / 1e6:,.1f} MB -> {n_out / 1e6:,.1f} MB "
          f"({n_in / max(n_out, 1):.2f}x)" + ("; sources deleted" if args.delete_source else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
