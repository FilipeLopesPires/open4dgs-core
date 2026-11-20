#!/usr/bin/env python3
"""Write the checkpoint copy that the export stage decodes and packages.

The source checkpoint (``trigs_ckpt.pt``) is never modified. The copy differs in
two ways:

1. It carries the run's training config. ``trigs-train-colmap`` saves no config in
   ``trigs_ckpt.pt``, so ``trigs-export-plys`` and ``make-trigs-timeline.py`` fall
   back to defaults. The deform settings that matter for export (above all
   ``temporal_active_threshold``, the temporal opacity threshold the run was
   trained with) are recovered, in order, from:

   - the checkpoint's own ``config`` entry (training snapshots have one),
   - the newest viewer snapshot of the run (``<run>/viewer_snapshots/latest.json``),
     whose ``config`` is the run's resolved ``TrigsTrainingConfig``.

   The frame count and frame rate come from the same place, then from
   ``<run>/run_args.json``, then from ``--n-frames``/``--fps``.

2. Rows whose base opacity ``sigmoid(opacities)`` is below ``--min-opacity`` are
   removed (default 0.005, 0 keeps every row). Such a row adds at most 0.5 % alpha
   to any pixel; on the 3 s ``professor`` run they are 9.3 % of the rows.

A JSON record of what was resolved and removed is written next to the output
(``<output>.json``).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Training-config fields that change how a checkpoint decodes (mirrors
# trigs.cli.export_plys._DEFORM_RELEVANT_FIELDS); printed for the record.
DEFORM_FIELDS = (
    "fps",
    "gauge_eps",
    "gauge_denom_clamp",
    "temporal_active_cull",
    "temporal_active_threshold",
    "temporal_active_cull_threshold",
    "temporal_active_extra_margin_s",
    "temporal_gamma_smooth_cull",
)


def config_as_dict(config: Any) -> dict[str, Any] | None:
    """A training config as a plain dict (dict, dataclass or object), else None."""
    if config is None:
        return None
    if isinstance(config, dict):
        return dict(config)
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return None


def _load(path: Path) -> dict[str, Any]:
    # mmap: tensors stay on disk until touched, so reading a snapshot's config
    # or a checkpoint's shapes costs no RAM.
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:  # legacy (non-zip) serialisation cannot be mapped
        return torch.load(path, map_location="cpu", weights_only=False)


def latest_viewer_snapshot(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    """The snapshot ``<run>/viewer_snapshots/latest.json`` points to, with the pointer."""
    pointer = run_dir / "viewer_snapshots" / "latest.json"
    if not pointer.is_file():
        return None
    try:
        info = json.loads(pointer.read_text())
    except (OSError, ValueError):
        return None
    snap = pointer.parent / str(info.get("file", ""))
    return (snap, info) if info.get("file") and snap.is_file() else None


def resolve_run_metadata(
    ckpt: dict[str, Any],
    run_dir: Path,
    n_rows: int,
    fallback_n_frames: int | None,
    fallback_fps: float | None,
) -> dict[str, Any]:
    """Training config, frame count and fps of the run, with where each came from."""
    out: dict[str, Any] = {
        "config": config_as_dict(ckpt.get("config")),
        "config_source": "checkpoint" if ckpt.get("config") is not None else None,
        "n_frames": ckpt.get("n_frames"),
        "n_frames_source": "checkpoint" if ckpt.get("n_frames") else None,
        "fps": ckpt.get("fps"),
        "fps_source": "checkpoint" if ckpt.get("fps") else None,
        "warnings": [],
    }

    if out["config"] is None or not out["n_frames"] or not out["fps"]:
        found = latest_viewer_snapshot(run_dir)
        if found is not None:
            snap_path, info = found
            snap_rows = (info.get("extra") or {}).get("n_rows")
            if snap_rows is not None and int(snap_rows) != n_rows:
                out["warnings"].append(
                    f"{snap_path.name} has {int(snap_rows):,} rows, the checkpoint "
                    f"{n_rows:,}: snapshot may be from another training of this run name"
                )
            snap = _load(snap_path)
            label = f"viewer_snapshot:{snap_path.name}"
            if out["config"] is None:
                out["config"] = config_as_dict(snap.get("config"))
                if out["config"] is not None:
                    out["config_source"] = label
            if not out["n_frames"] and snap.get("n_frames"):
                out["n_frames"], out["n_frames_source"] = int(snap["n_frames"]), label
            if not out["fps"] and snap.get("fps"):
                out["fps"], out["fps_source"] = float(snap["fps"]), label
            del snap

    if not out["fps"] and out["config"] and out["config"].get("fps"):
        out["fps"], out["fps_source"] = float(out["config"]["fps"]), "config"

    if not out["n_frames"]:
        run_args = run_dir / "run_args.json"
        try:
            ra = json.loads(run_args.read_text())
            n = int(ra["end_frame"]) - int(ra["start_frame"]) + 1
            if n > 0:
                out["n_frames"], out["n_frames_source"] = n, "run_args.json"
        except (OSError, ValueError, KeyError, TypeError):
            pass

    if not out["n_frames"] and fallback_n_frames:
        out["n_frames"], out["n_frames_source"] = int(fallback_n_frames), "command_line"
    elif out["n_frames"] and fallback_n_frames and int(fallback_n_frames) != int(out["n_frames"]):
        out["warnings"].append(
            f"--n-frames {fallback_n_frames} ignored: the run was trained on "
            f"{out['n_frames']} frames ({out['n_frames_source']})"
        )
    if not out["fps"] and fallback_fps:
        out["fps"], out["fps_source"] = float(fallback_fps), "command_line"
    elif out["fps"] and fallback_fps and abs(float(fallback_fps) - float(out["fps"])) > 1e-6:
        out["warnings"].append(
            f"--fps {fallback_fps:g} ignored: the run was trained at {float(out['fps']):g} fps "
            f"({out['fps_source']})"
        )
    if out["config"] is None:
        out["warnings"].append(
            "no training config found (checkpoint has none, no viewer snapshot): "
            "exporters will decode with trainer defaults"
        )
    return out


def opacity_keep_mask(opacities: torch.Tensor, min_opacity: float) -> torch.Tensor:
    """Rows to keep: base opacity sigmoid(logit) >= min_opacity (all rows when <= 0)."""
    flat = opacities.detach().reshape(-1).float()
    if min_opacity <= 0.0:
        return torch.ones_like(flat, dtype=torch.bool)
    return torch.sigmoid(flat) >= float(min_opacity)


def slice_rows(value: Any, keep: torch.Tensor, n_rows: int) -> Any:
    """Apply a row mask to every tensor with ``n_rows`` leading rows, recursing into dicts."""
    if torch.is_tensor(value):
        if value.dim() >= 1 and int(value.shape[0]) == n_rows:
            return value[keep].clone()
        return value
    if isinstance(value, dict):
        return {k: slice_rows(v, keep, n_rows) for k, v in value.items()}
    return value


def prepare(
    checkpoint: Path,
    output: Path,
    *,
    min_opacity: float,
    run_dir: Path | None = None,
    n_frames: int | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    ckpt = _load(checkpoint)
    params = ckpt.get("params")
    if not isinstance(params, dict) or "opacities" not in params or "means" not in params:
        raise SystemExit(f"{checkpoint}: not a TriGS checkpoint (no params/opacities)")
    n_rows = int(params["means"].shape[0])
    run_dir = Path(run_dir) if run_dir is not None else checkpoint.parent

    meta = resolve_run_metadata(ckpt, run_dir, n_rows, n_frames, fps)
    keep = opacity_keep_mask(params["opacities"], min_opacity)
    kept = int(keep.sum())
    if kept == 0:
        raise SystemExit(f"--min-opacity {min_opacity:g} would remove every row")

    out = {k: v for k, v in ckpt.items()}
    for key in ("params", "temporal_params", "motion_params"):
        if key in out:
            out[key] = slice_rows(out[key], keep, n_rows)
    if meta["config"] is not None:
        out["config"] = meta["config"]
    if meta["n_frames"]:
        out["n_frames"] = int(meta["n_frames"])
    if meta["fps"]:
        out["fps"] = float(meta["fps"])

    config = meta["config"] or {}
    record = {
        "source_checkpoint": str(checkpoint),
        "output": str(output),
        "written_unix": time.time(),
        "rows_in": n_rows,
        "rows_out": kept,
        "rows_removed": n_rows - kept,
        "min_opacity": float(min_opacity),
        "config_source": meta["config_source"],
        "n_frames": meta["n_frames"],
        "n_frames_source": meta["n_frames_source"],
        "fps": meta["fps"],
        "fps_source": meta["fps_source"],
        "deform_config": {k: config.get(k) for k in DEFORM_FIELDS if k in config},
        "warnings": meta["warnings"],
    }
    out["export_prep"] = {k: v for k, v in record.items() if k != "output"}

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    torch.save(out, tmp)
    tmp.replace(output)
    Path(str(output) + ".json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True, type=Path, help="Source trigs_ckpt.pt (read only)")
    ap.add_argument("--output", required=True, type=Path, help="Checkpoint copy to write")
    ap.add_argument("--min-opacity", type=float, default=0.005,
                    help="Drop rows with sigmoid(opacity) below this; 0 keeps all rows. Default: 0.005")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Run folder with viewer_snapshots\\ and run_args.json. Default: the checkpoint's folder")
    ap.add_argument("--n-frames", type=int, default=None,
                    help="Frame count, used only when the run records none")
    ap.add_argument("--fps", type=float, default=None,
                    help="Frame rate, used only when the run records none")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if args.min_opacity < 0.0 or args.min_opacity >= 1.0:
        raise SystemExit("--min-opacity must be in [0, 1)")
    if args.output.resolve() == args.checkpoint.resolve():
        raise SystemExit("--output must differ from --checkpoint (the source is never modified)")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite")

    rec = prepare(
        args.checkpoint, args.output, min_opacity=args.min_opacity,
        run_dir=args.run_dir, n_frames=args.n_frames, fps=args.fps,
    )
    print(f"Export checkpoint: {args.output}")
    print(f"  source: {rec['source_checkpoint']}")
    pct = 100.0 * rec["rows_removed"] / max(rec["rows_in"], 1)
    print(f"  rows: {rec['rows_in']:,} -> {rec['rows_out']:,} "
          f"({rec['rows_removed']:,} = {pct:.1f}% below opacity {rec['min_opacity']:g})")
    print(f"  training config: {rec['config_source'] or 'none (trainer defaults)'}")
    for k, v in rec["deform_config"].items():
        print(f"    {k} = {v}")
    print(f"  frames: {rec['n_frames']} ({rec['n_frames_source']}), fps: {rec['fps']} ({rec['fps_source']})")
    for w in rec["warnings"]:
        print(f"  WARNING: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
