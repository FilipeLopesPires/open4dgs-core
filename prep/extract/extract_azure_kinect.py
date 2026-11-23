#!/usr/bin/env python3
"""Import multi-device Azure Kinect (k4arecorder .mkv) captures into the Open4DGS layout.

Two subcommands, both re-runnable:

``inspect``  Audit a capture without writing frames: per-device metadata, factory
             calibration, timestamp regularity / dropped frames, clock groups, the
             frame-alignment solve (validated against timestamp ground truth), the
             common frame window, plus QA images:
               * ``sync_sheet.jpg``   - every camera at a few logical frames, rotated
                                        and undistorted, for eyeballing sync/rotation;
               * ``depth_overlay.jpg`` - depth projected onto colour per camera
                                        (checks the factory depth->colour calibration).

``extract``  Decode the common window, undistort colour to PINHOLE, rotate, and write

                 images/<0001..N>/<000001..>.jpg     undistorted colour
                 depth/<0001..N>/<000001..>.png      raw 16-bit sensor depth (mm)
                 k4a_cameras.json                     per-camera intrinsics, depth
                                                      calibration, device metadata
                 sync.json                            per-camera offset (s), PLUS convention
                 frame_map.json                       logical frame -> source packets
                 k4a_report/inspect.json              same report as ``inspect``

Extrinsics are *not* stored in the recordings; run
``prep/align/calibrate_azure_kinect.py`` next to produce ``sparse/0`` and
``reference_export.json``.

Example::

    python prep/extract/extract_azure_kinect.py inspect \
        --input-dir datasets/<capture> --output-dir datasets/<capture>/o4d --rotate l
    python prep/extract/extract_azure_kinect.py extract \
        --input-dir datasets/<capture> --output-dir datasets/<capture>/o4d --rotate l
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from k4a_mkv import Intrinsics, K4ARecording, iter_frames, open_recording, read_frame  # noqa: E402
from k4a_sync import CameraTiming, common_window, solve_timing  # noqa: E402

ROTATIONS = ("n", "l", "r", "u")  # none / 90 deg counter-clockwise / clockwise / 180


# --------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------
def rotation_matrix_for_image(rotation: str) -> np.ndarray:
    """Camera-frame rotation applied when the image is rotated (matches extract_4dv_sequence)."""
    return np.array(
        {
            "n": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "l": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            "r": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            "u": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        }[rotation],
        dtype=np.float64,
    )


def rotate_K(K: np.ndarray, width: int, height: int, rotation: str) -> tuple[np.ndarray, int, int]:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if rotation == "n":
        return K.copy(), width, height
    if rotation == "l":
        return np.array([[fy, 0, cy], [0, fx, width - 1.0 - cx], [0, 0, 1]]), height, width
    if rotation == "r":
        return np.array([[fy, 0, height - 1.0 - cy], [0, fx, cx], [0, 0, 1]]), height, width
    return np.array([[fx, 0, width - 1.0 - cx], [0, fy, height - 1.0 - cy], [0, 0, 1]]), width, height


def rotate_image(img: np.ndarray, rotation: str) -> np.ndarray:
    if rotation == "n":
        return img
    code = {
        "l": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "r": cv2.ROTATE_90_CLOCKWISE,
        "u": cv2.ROTATE_180,
    }[rotation]
    return cv2.rotate(img, code)


class OutputCamera:
    """Undistort + rotate pipeline for one device, plus the resulting pinhole model."""

    def __init__(self, intr: Intrinsics, rotation: str, alpha: float):
        size = (intr.width, intr.height)
        K_new, _ = cv2.getOptimalNewCameraMatrix(intr.K, intr.dist, size, alpha, size, centerPrincipalPoint=False)
        self.K_undist = K_new
        self.map1, self.map2 = cv2.initUndistortRectifyMap(intr.K, intr.dist, None, K_new, size, cv2.CV_16SC2)
        self.rotation = rotation
        self.K, self.width, self.height = rotate_K(K_new, intr.width, intr.height, rotation)
        self.R_rot = rotation_matrix_for_image(rotation)

    def apply(self, img: np.ndarray) -> np.ndarray:
        und = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return rotate_image(und, self.rotation)


def depth_points(depth_mm: np.ndarray, d_intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a raw depth image to depth-camera 3D points (metres). Returns (pts, pixel_idx)."""
    v, u = np.nonzero(depth_mm > 0)
    z = depth_mm[v, u].astype(np.float64) / 1000.0
    uv = np.stack([u, v], 1).astype(np.float64)[:, None, :]
    xy = cv2.undistortPoints(uv, d_intr.K, d_intr.dist)[:, 0, :]
    return np.c_[xy * z[:, None], z], np.stack([v, u], 1)


def render_depth_in_color(
    pts_depth: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Z-buffered sparse depth (metres) of depth-camera points seen by a pinhole colour camera."""
    Xc = pts_depth @ R.T + t
    z = Xc[:, 2]
    keep = z > 0.1
    Xc, z = Xc[keep], z[keep]
    u = np.round(K[0, 0] * Xc[:, 0] / z + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * Xc[:, 1] / z + K[1, 2]).astype(np.int64)
    ok = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[ok], v[ok], z[ok]
    out = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(out, (v, u), z.astype(np.float32))
    out[~np.isfinite(out)] = 0
    return out


# --------------------------------------------------------------------------------------
# Discovery / shared setup
# --------------------------------------------------------------------------------------
def discover(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise SystemExit(f"no recordings matching {pattern} in {input_dir}")
    return files


def parse_overrides(items: list[str] | None, kind: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--{kind} expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_capture(args) -> tuple[list[K4ARecording], int, dict[str, str]]:
    files = discover(Path(args.input_dir), args.pattern)
    print(f"Found {len(files)} recordings")
    recs = []
    for f in files:
        r = open_recording(f)
        recs.append(r)
        print(
            f"  {r.name:24s} {r.sync_mode:11s} serial={r.serial} colour={r.color_mode} depth={r.depth_mode} "
            f"frames={len(r.color_pts_ns)} sub_delay={r.subordinate_delay_ns / 1e3:.0f}us"
        )
    fps = {round(r.fps, 3) for r in recs}
    if len(fps) != 1:
        raise SystemExit(f"recordings disagree on frame rate: {sorted(fps)}")
    masters = [i for i, r in enumerate(recs) if r.sync_mode == "MASTER"]
    if args.reference:
        names = [r.name for r in recs]
        if args.reference not in names:
            raise SystemExit(f"--reference {args.reference} not among {names}")
        reference = names.index(args.reference)
    elif len(masters) == 1:
        reference = masters[0]
    else:
        reference = 0
        if not masters:
            print("WARNING: no MASTER device; recordings may not be hardware-synchronised")
    rot_over = parse_overrides(args.rotate_camera, "rotate-camera")
    rotations = {}
    for r in recs:
        rot = rot_over.get(r.name, args.rotate)
        if rot not in ROTATIONS:
            raise SystemExit(f"rotation for {r.name} must be one of {ROTATIONS}")
        rotations[r.name] = rot
    return recs, reference, rotations


def timestamp_audit(r: K4ARecording) -> dict:
    period = 1e9 / r.fps
    out = {}
    for kind, ts in (("color", r.color_device_ns), ("depth", r.depth_device_ns)):
        if ts is None or len(ts) < 2:
            continue
        dt = np.diff(ts.astype(np.float64))
        slots = np.round((ts - ts[0]) / period).astype(int)
        missing = int(slots[-1] + 1 - len(np.unique(slots)))
        out[kind] = {
            "packets": int(len(ts)),
            "first_device_ms": float(ts[0] / 1e6),
            "last_device_ms": float(ts[-1] / 1e6),
            "interval_median_ms": float(np.median(dt) / 1e6),
            "interval_max_ms": float(dt.max() / 1e6),
            "dropped_frames": missing,
            "irregular_intervals": int((np.abs(dt - period) > 0.25 * period).sum()),
        }
    return out


def solve_or_load_timing(args, recs, reference, report_dir: Path):
    timing_path = report_dir / "timing.json"
    if args.timing_json:
        timing_path = Path(args.timing_json)
    if timing_path.exists() and not args.resolve_timing:
        data = json.loads(timing_path.read_text(encoding="utf-8"))
        names = [r.name for r in recs]
        if sorted(data["cameras"]) != sorted(names):
            raise SystemExit(f"{timing_path} does not match these recordings; pass --resolve-timing")
        timings = [
            CameraTiming(n, data["cameras"][n]["lag"], data["cameras"][n]["offset_ms"] / 1e3,
                         data["cameras"][n]["method"], data["cameras"][n]["clock_group"])
            for n in names
        ]
        print(f"Reusing timing solution from {timing_path} (pass --resolve-timing to recompute)")
        return timings, data
    print("Solving frame alignment ...")
    manual = {k: int(v) for k, v in parse_overrides(args.manual_lag, "manual-lag").items()}
    timings, rep = solve_timing(
        recs, reference, skip_startup=args.skip_startup_frames, max_lag=args.max_lag,
        manual_lags=manual, run_visual=not args.no_visual_sync,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    return timings, rep


def print_timing(timings: list[CameraTiming], rep: dict) -> None:
    print("Frame alignment (logical frame i -> source slot i+lag; actual = logical + offset):")
    for t in timings:
        vis = "" if t.visual_lag is None else f" visual_lag={t.visual_lag:+.2f}"
        print(f"  {t.name:24s} lag={t.lag:+d} offset={t.offset_s * 1e3:+8.3f} ms  [{t.method}]{vis}")
        for n in t.notes:
            print(f"      {n}")
    v = rep.get("visual_validation_max_error_frames")
    if v is not None:
        flag = "OK" if v < 0.5 else "CHECK - visual sync disagrees with timestamps"
        print(f"  visual-sync validation vs timestamp truth: max error {v:.2f} frames ({flag})")


# --------------------------------------------------------------------------------------
# QA sheets
# --------------------------------------------------------------------------------------
def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
    cv2.putText(img, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    return img


def _grid(tiles: list[np.ndarray], cols: int, tile_w: int) -> np.ndarray:
    scaled = [cv2.resize(t, (tile_w, int(round(t.shape[0] * tile_w / t.shape[1])))) for t in tiles]
    h = max(t.shape[0] for t in scaled)
    scaled = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in scaled]
    while len(scaled) % cols:
        scaled.append(np.zeros_like(scaled[0]))
    rows = [np.hstack(scaled[i : i + cols]) for i in range(0, len(scaled), cols)]
    return np.vstack(rows)


def qa_sheets(recs, timings, slots, first, last, rotations, alpha, report_dir: Path) -> None:
    cams = [OutputCamera(r.color_intrinsics(), rotations[r.name], alpha) for r in recs]
    # sync sheet: 3 logical frames x cameras
    picks = [first + (last - first) * f // 4 for f in (1, 2, 3)]
    rows = []
    for k in picks:
        tiles = []
        for r, t, s, cam in zip(recs, timings, slots, cams):
            img = cam.apply(read_frame(r, "color", s[k + t.lag]))
            tiles.append(_label(img, f"{r.name} L{k - first + 1}"))
        rows.append(_grid(tiles, len(tiles), 320))
    cv2.imwrite(str(report_dir / "sync_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # depth overlay at the middle frame
    k = picks[1]
    tiles = []
    for r, t, s, cam in zip(recs, timings, slots, cams):
        if r.depth_stream is None:
            continue
        pkt = s[k + t.lag]
        color = read_frame(r, "color", pkt)
        dpkt = nearest_depth_packet(r, pkt)
        if dpkt is None:
            continue
        depth = read_frame(r, "depth", dpkt)
        pts, _ = depth_points(depth, r.depth_intrinsics())
        R, tt = r.depth_to_color()
        R = cam.R_rot @ R
        tt = cam.R_rot @ tt
        dm = render_depth_in_color(pts, R, tt, cam.K, cam.width, cam.height)
        dm = cv2.dilate(dm, np.ones((5, 5), np.uint8))
        img = cam.apply(color)
        valid = dm > 0
        lo, hi = np.percentile(dm[valid], [2, 98]) if valid.any() else (0.5, 5.0)
        cmap = cv2.applyColorMap(np.clip((dm - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        # checkerboard blend makes misalignment of edges easy to see
        yy, xx = np.indices(dm.shape)
        checker = ((yy // 64 + xx // 64) % 2 == 0) & valid
        img[checker] = (0.35 * img[checker] + 0.65 * cmap[checker]).astype(np.uint8)
        tiles.append(_label(img, r.name))
    if tiles:
        cv2.imwrite(str(report_dir / "depth_overlay.jpg"), _grid(tiles, min(3, len(tiles)), 540), [cv2.IMWRITE_JPEG_QUALITY, 88])


def nearest_depth_packet(r: K4ARecording, color_pkt: int, tol_ms: float = 20.0) -> int | None:
    if r.depth_device_ns is None:
        return None
    tc = r.color_device_ns[color_pkt]
    j = int(np.argmin(np.abs(r.depth_device_ns - tc)))
    if abs(int(r.depth_device_ns[j]) - int(tc)) > tol_ms * 1e6:
        return None
    return j


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------
def build_report(args, recs, reference, rotations, timings, rep, first, last) -> dict:
    return {
        "tool": "extract_azure_kinect.py",
        "input_dir": str(Path(args.input_dir).resolve()),
        "fps": recs[reference].fps,
        "reference": recs[reference].name,
        "rotations": rotations,
        "undistort_alpha": args.undistort_alpha,
        "skip_startup_frames": args.skip_startup_frames,
        "common_window": {"first_logical": first, "last_logical": last, "frames": last - first + 1},
        "recordings": {
            r.name: {
                "path": str(r.path),
                "serial": r.serial,
                "sync_mode": r.sync_mode,
                "subordinate_delay_ns": r.subordinate_delay_ns,
                "start_offset_ns": r.start_offset_ns,
                "color_mode": r.color_mode,
                "depth_mode": r.depth_mode,
                "color_intrinsics": r.color_intrinsics().to_dict(),
                "depth_intrinsics": r.depth_intrinsics().to_dict() if r.depth_stream is not None else None,
                "timestamps": timestamp_audit(r),
            }
            for r in recs
        },
        "timing": rep,
    }


def cmd_inspect(args) -> int:
    out = Path(args.output_dir)
    report_dir = out / "k4a_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    recs, reference, rotations = load_capture(args)
    timings, rep = solve_or_load_timing(args, recs, reference, report_dir)
    print_timing(timings, rep)
    first, last, slots = common_window(recs, timings, args.skip_startup_frames)
    print(f"Common window: logical {first}..{last} ({last - first + 1} frames, {(last - first + 1) / recs[0].fps:.2f} s)")
    report = build_report(args, recs, reference, rotations, timings, rep, first, last)
    (report_dir / "inspect.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.no_qa:
        print("Writing QA sheets ...")
        qa_sheets(recs, timings, slots, first, last, rotations, args.undistort_alpha, report_dir)
    print(f"Report: {report_dir / 'inspect.json'}")
    for r in recs:
        a = report["recordings"][r.name]["timestamps"].get("color", {})
        if a.get("dropped_frames"):
            print(f"  NOTE {r.name}: {a['dropped_frames']} dropped colour frames (handled by slot mapping)")
    return 0


def _extract_camera(job: dict) -> dict:
    """Worker: decode one device's colour (and depth) for the selected logical frames."""
    r = open_recording(Path(job["path"]), name=job["name"])
    cam = OutputCamera(r.color_intrinsics(), job["rotation"], job["alpha"])
    img_dir = Path(job["images_dir"])
    img_dir.mkdir(parents=True, exist_ok=True)
    pkt_to_out = {int(k): v for k, v in job["color_packets"].items()}
    lo, hi = min(pkt_to_out), max(pkt_to_out)
    written = 0
    w, h = r.color_size
    for pkt, frame in iter_frames(r.path, r.color_stream, w, h, "bgr24", first=lo, count=hi - lo + 1):
        name = pkt_to_out.get(pkt)
        if name is None:
            continue
        dst = img_dir / name
        if job["skip_existing"] and dst.exists():
            continue
        cv2.imwrite(str(dst), cam.apply(frame), [cv2.IMWRITE_JPEG_QUALITY, job["jpeg_quality"]])
        written += 1
    dwritten = 0
    if job["depth_dir"] and r.depth_stream is not None and job["depth_packets"]:
        ddir = Path(job["depth_dir"])
        ddir.mkdir(parents=True, exist_ok=True)
        dmap = {int(k): v for k, v in job["depth_packets"].items()}
        dlo, dhi = min(dmap), max(dmap)
        dw, dh = r.depth_size
        for pkt, frame in iter_frames(r.path, r.depth_stream, dw, dh, "gray16be", first=dlo, count=dhi - dlo + 1):
            name = dmap.get(pkt)
            if name is None:
                continue
            dst = ddir / name
            if job["skip_existing"] and dst.exists():
                continue
            cv2.imwrite(str(dst), frame)
            dwritten += 1
    return {"name": r.name, "color_written": written, "depth_written": dwritten}


def cmd_extract(args) -> int:
    out = Path(args.output_dir)
    report_dir = out / "k4a_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    recs, reference, rotations = load_capture(args)
    timings, rep = solve_or_load_timing(args, recs, reference, report_dir)
    print_timing(timings, rep)
    first, last, slots = common_window(recs, timings, args.skip_startup_frames)
    start = first + args.start_frame
    n = last - start + 1 if args.num_frames is None else args.num_frames
    if start > last or start + n - 1 > last or n <= 0:
        raise SystemExit(f"requested window exceeds the common window {first}..{last} ({last - first + 1} frames)")
    logical = list(range(start, start + n))
    print(f"Extracting {n} frames (logical {start}..{start + n - 1}) from window {first}..{last}")

    images_root = out / "images"
    if images_root.exists() and any(images_root.iterdir()) and not (args.overwrite or args.skip_existing):
        raise SystemExit(f"{images_root} is not empty; pass --skip-existing or --overwrite")

    folders = [f"{i + 1:04d}" for i in range(len(recs))]
    jobs, frame_map, cameras_json, sync = [], {}, {}, {}
    for folder, r, t, s in zip(folders, recs, timings, slots):
        color_packets, depth_packets, rows = {}, {}, []
        for j, k in enumerate(logical):
            fname = f"{j + 1:06d}"
            pkt = s[k + t.lag]
            color_packets[pkt] = f"{fname}.jpg"
            dpkt = nearest_depth_packet(r, pkt) if not args.no_depth else None
            if dpkt is not None:
                depth_packets[dpkt] = f"{fname}.png"
            rows.append({"frame": j + 1, "color_packet": pkt, "color_device_ns": int(r.color_device_ns[pkt]),
                         "depth_packet": dpkt})
        frame_map[folder] = rows
        jobs.append({
            "path": str(r.path), "name": r.name, "rotation": rotations[r.name], "alpha": args.undistort_alpha,
            "images_dir": str(images_root / folder), "color_packets": color_packets,
            "depth_dir": None if args.no_depth else str(out / "depth" / folder), "depth_packets": depth_packets,
            "skip_existing": args.skip_existing, "jpeg_quality": args.jpeg_quality,
        })
        cam = OutputCamera(r.color_intrinsics(), rotations[r.name], args.undistort_alpha)
        R_dc, t_dc = r.depth_to_color()
        cameras_json[folder] = {
            "source": r.name,
            "serial": r.serial,
            "sync_mode": r.sync_mode,
            "rotation": rotations[r.name],
            "width": cam.width,
            "height": cam.height,
            "K": cam.K.tolist(),
            "camera_model": "PINHOLE",
            "params": [cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]],
            "raw_color_intrinsics": r.color_intrinsics().to_dict(),
            "depth_intrinsics": r.depth_intrinsics().to_dict() if r.depth_stream is not None else None,
            # Maps depth-camera points (metres) into the *output* (undistorted, rotated) colour camera.
            "depth_to_color_R": (cam.R_rot @ R_dc).tolist(),
            "depth_to_color_t": (cam.R_rot @ t_dc).tolist(),
            "timing": {"lag": t.lag, "offset_s": t.offset_s, "method": t.method},
        }
        sync[folder] = t.offset_s

    t0 = time.time()
    workers = max(1, min(args.workers, len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_extract_camera, jobs):
            print(f"  {res['name']}: {res['color_written']} colour, {res['depth_written']} depth frames written")
    print(f"Decoded in {time.time() - t0:.1f} s")

    meta = {
        "schema": "open4dgs-azure-kinect-v1",
        "fps": recs[reference].fps,
        "num_frames": n,
        "frame_pattern": "{frame:06d}",
        "first_frame": 1,
        "depth_units": "millimetres (raw sensor image; use depth_intrinsics + depth_to_color_*)",
        "cameras": cameras_json,
    }
    (out / "k4a_cameras.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out / "sync.json").write_text(json.dumps(sync, indent=2) + "\n", encoding="utf-8")
    (out / "frame_map.json").write_text(json.dumps({"logical_start": start, "cameras": frame_map}, indent=2) + "\n", encoding="utf-8")
    report = build_report(args, recs, reference, rotations, timings, rep, first, last)
    report["extracted"] = {"logical_start": start, "num_frames": n, "folders": dict(zip(folders, [r.name for r in recs]))}
    (report_dir / "inspect.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.no_qa and not (report_dir / "sync_sheet.jpg").exists():
        qa_sheets(recs, timings, slots, first, last, rotations, args.undistort_alpha, report_dir)
    print(f"Wrote {out / 'k4a_cameras.json'}, {out / 'sync.json'}, {out / 'frame_map.json'}")
    print("Next: python prep/align/calibrate_azure_kinect.py --data-dir", out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--input-dir", required=True, help="directory holding the .mkv recordings")
        sp.add_argument("--output-dir", required=True, help="capture data directory to create/update")
        sp.add_argument("--pattern", default="*.mkv", help="recording glob; sorted order sets camera folders 0001..")
        sp.add_argument("--reference", help="recording name (file stem) defining logical time (default: MASTER)")
        sp.add_argument("--rotate", default="n", choices=ROTATIONS,
                        help="image rotation for all cameras: n=none, l=90 CCW, r=90 CW, u=180")
        sp.add_argument("--rotate-camera", action="append", metavar="NAME=ROT", help="per-recording rotation override")
        sp.add_argument("--undistort-alpha", type=float, default=0.0,
                        help="cv2.getOptimalNewCameraMatrix alpha: 0 keeps only valid pixels, 1 keeps all")
        sp.add_argument("--skip-startup-frames", type=int, default=2,
                        help="ignore the first N packets per device (sync lock-in transients)")
        sp.add_argument("--max-lag", type=int, default=30, help="visual sync search range (frames)")
        sp.add_argument("--manual-lag", action="append", metavar="NAME=LAG",
                        help="force integer slot lag for a recording on a separate clock")
        sp.add_argument("--no-visual-sync", action="store_true", help="timestamps only (requires --manual-lag for other clocks)")
        sp.add_argument("--timing-json", help="reuse this timing solution instead of k4a_report/timing.json")
        sp.add_argument("--resolve-timing", action="store_true", help="recompute timing even if a saved solution exists")
        sp.add_argument("--no-qa", action="store_true", help="skip QA image sheets")

    sp = sub.add_parser("inspect", help="audit the capture and write QA reports (no frame export)")
    common(sp)
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("extract", help="export undistorted frames, depth and metadata")
    common(sp)
    sp.add_argument("--start-frame", type=int, default=0, help="offset into the common window")
    sp.add_argument("--num-frames", type=int, help="frames to export (default: rest of the common window)")
    sp.add_argument("--no-depth", action="store_true", help="do not export depth frames")
    sp.add_argument("--jpeg-quality", type=int, default=95)
    sp.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    sp.add_argument("--skip-existing", action="store_true")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_extract)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
