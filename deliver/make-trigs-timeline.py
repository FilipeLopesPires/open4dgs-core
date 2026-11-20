#!/usr/bin/env python3
"""
TRiGS Runtime Binary Packager — authoring .trigs / checkpoint -> production .trigsbin.

Version: 2026-09-16.runtime-bin-v2

This script produces a single engine-neutral runtime file:

  [64-byte fixed header][UTF-8 JSON runtime manifest][padding][aligned raw tensor payloads]

The JSON manifest describes timeline/clip scheduling and each tensor as a typed
view into payload buffer 0. Tensor byte offsets are RELATIVE TO payloadOffset,
so loaders can seek to:

    absolute_byte_offset = header.payloadOffset + tensor.byteOffset

Goals:
  - Keep .trigs as the authoring/interchange archive.
  - Use .trigsbin as the renderer/runtime cache for Python, Unity, Unreal, etc.
  - Preserve TRiGS static/dynamic split without baking in a Python-viewer-only layout.
  - Store tensors as SoA typed buffers with explicit dtype, shape, semantic, and domains.

Examples:

  # Convert existing authoring archive to runtime binary.
  python make_trigsbin_prod.py from-trigs \
    --input scene.trigs \
    --output scene.trigsbin \
    --overwrite

  # Build runtime binary directly from trainer checkpoints with manual schedule.
  python make_trigsbin_prod.py from-checkpoints \
    --output test-two-clip.trigsbin \
    --frame-rate 60 \
    --checkpoint results-clip-0000/trigs_ckpt.pt --range 0:360:0:300 \
    --checkpoint results-clip-0001/trigs_ckpt.pt --range 240:600:360:600 \
    --fade-frames 60 \
    --tensor-dtype motion32 \
    --overwrite

  # Split one long-sequence checkpoint using its saved trainer windows.
  python make_trigs_timeline.py from-checkpoint \
    --checkpoint results-long/trigs_ckpt.pt \
    --output long-sequence.trigsbin \
    --frame-rate 60 \
    --tensor-dtype motion32 \
    --overwrite

  # Inspect an existing runtime binary header + manifest summary.
  python make_trigsbin_prod.py inspect --input scene.trigsbin
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import copy
import hashlib
import json
import math
import os
import struct
import sys
import time
import zipfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

VERSION = "2026-09-16.runtime-bin-v2"
RUNTIME_FORMAT = "trigs.runtime"
RUNTIME_VERSION = 1
MAGIC = b"TRIGSBIN"  # exactly 8 bytes
HEADER_STRUCT = struct.Struct("<8sIIQQQQQQ")
HEADER_BYTES = HEADER_STRUCT.size  # 64 bytes
DEFAULT_ALIGNMENT = 256
MANIFEST_NAME = "manifest.json"
SUPPORTED_DTYPES = ("float32", "float16", "mixed", "motion32")

REQUIRED_KEYS = (
    "means",
    "scales",
    "quats",
    "opacities",
    "sh0",
    "shN",
    "times",
    "log_durations",
    "anchors",
    "zeta_base_r",
    "zeta_base_t",
    "zeta_c0_r",
    "zeta_c0_t",
    "zeta_c1_r",
    "zeta_c1_t",
    "zeta_c2_r",
    "zeta_c2_t",
)
STATIC_KEYS = (
    "means",
    "scales",
    "quats",
    "opacities",
    "sh0",
    "shN",
)
OPTIONAL_FLAG_KEYS = ("is_static", "dynamic_protect")
TEMPORAL_FLOAT32_KEYS = {"times", "log_durations"}
MOTION_FLOAT32_KEYS = {
    "means",
    "quats",
    "times",
    "log_durations",
    "anchors",
    "zeta_base_r",
    "zeta_base_t",
    "zeta_c0_r",
    "zeta_c0_t",
    "zeta_c1_r",
    "zeta_c1_t",
    "zeta_c2_r",
    "zeta_c2_t",
}

DTYPE_NBYTES = {
    "float16": 2,
    "f16": 2,
    "float32": 4,
    "f32": 4,
    "uint8": 1,
    "u8": 1,
    "int32": 4,
    "i32": 4,
    "uint32": 4,
    "u32": 4,
}


@dataclass(frozen=True)
class ClipSchedule:
    index: int
    train_start: int
    train_end: int
    core_start: int
    core_end: int

    @property
    def train_frames(self) -> int:
        return self.train_end - self.train_start

    @property
    def core_frames(self) -> int:
        return self.core_end - self.core_start

    @property
    def local_core_start(self) -> int:
        return self.core_start - self.train_start

    @property
    def local_core_end(self) -> int:
        return self.local_core_start + self.core_frames


@dataclass
class TensorPayload:
    logical_path: str
    dtype: str
    shape: list[int]
    data: bytes
    spec_ref: dict[str, Any]
    clip_id: str | None = None
    group: str | None = None
    name: str | None = None


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


def canonical_dtype(dtype: str) -> str:
    d = str(dtype).lower()
    if d in ("f16", "float16"):
        return "float16"
    if d in ("f32", "float32"):
        return "float32"
    if d in ("u8", "uint8"):
        return "uint8"
    if d in ("i32", "int32"):
        return "int32"
    if d in ("u32", "uint32"):
        return "uint32"
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def dtype_ext(dtype: str) -> str:
    d = canonical_dtype(dtype)
    if d == "float32":
        return "f32"
    if d == "float16":
        return "f16"
    if d == "uint8":
        return "u8"
    if d == "int32":
        return "i32"
    if d == "uint32":
        return "u32"
    raise ValueError(dtype)


def dtype_nbytes(dtype: str) -> int:
    return DTYPE_NBYTES[canonical_dtype(dtype)]


def expected_nbytes(dtype: str, shape: Iterable[int]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return int(n) * dtype_nbytes(dtype)


def ensure_little_endian(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype.byteorder in ("<", "=") and sys.byteorder == "little":
        return arr
    return arr.byteswap().view(arr.dtype.newbyteorder("<"))


def encode_array(array: np.ndarray, dtype: str) -> bytes:
    d = canonical_dtype(dtype)
    if d == "float32":
        return ensure_little_endian(np.asarray(array, dtype=np.float32)).tobytes(order="C")
    if d == "float16":
        return ensure_little_endian(np.asarray(array, dtype=np.float16)).tobytes(order="C")
    if d == "uint8":
        return ensure_little_endian(np.asarray(array, dtype=np.uint8)).tobytes(order="C")
    if d == "int32":
        return ensure_little_endian(np.asarray(array, dtype=np.int32)).tobytes(order="C")
    if d == "uint32":
        return ensure_little_endian(np.asarray(array, dtype=np.uint32)).tobytes(order="C")
    raise ValueError(dtype)


def tensor_semantic(name: str, group: str) -> dict[str, Any]:
    """Renderer-facing semantics. Kept stable and engine-neutral."""
    base: dict[str, Any] = {
        "domain": group,
        "layout": "SoA",
        "rowMajor": True,
    }
    if name == "means":
        base.update({"role": "gaussian.position", "components": "xyz", "domainValue": "world"})
    elif name == "scales":
        base.update({"role": "gaussian.scale", "components": "xyz", "valueDomain": "log"})
    elif name == "quats":
        base.update({"role": "gaussian.orientation", "components": "xyzw", "normalizedAtRuntime": True})
    elif name == "opacities":
        base.update({"role": "gaussian.opacity", "valueDomain": "logit", "activation": "sigmoid"})
    elif name == "sh0":
        base.update({"role": "gaussian.sh.dc", "coefficients": 1, "channelOrder": "rgb"})
    elif name == "shN":
        base.update({"role": "gaussian.sh.rest", "channelOrder": "rgb"})
    elif name == "times":
        base.update({"role": "motion.center_time", "unit": "seconds"})
    elif name == "log_durations":
        base.update({"role": "motion.log_duration", "unit": "seconds_log"})
    elif name == "anchors":
        base.update({"role": "motion.anchor", "components": "xyz", "domainValue": "world"})
    elif name.startswith("zeta_base") or name.startswith("zeta_c"):
        base.update({"role": "motion.se3_bezier_parameter", "model": "trigs.se3_bezier_v1"})
    elif name in OPTIONAL_FLAG_KEYS:
        base.update({"role": f"row_flag.{name}", "valueDomain": "boolean_uint8"})
    else:
        base.update({"role": f"tensor.{name}"})
    return base


class RuntimeBinaryBuilder:
    def __init__(self, alignment: int = DEFAULT_ALIGNMENT, include_hashes: bool = False):
        self.alignment = int(alignment)
        if self.alignment < 16:
            raise ValueError("alignment should be at least 16 bytes")
        self.include_hashes = bool(include_hashes)
        self.tensors: list[TensorPayload] = []

    def add_tensor(
        self,
        *,
        logical_path: str,
        dtype: str,
        shape: Iterable[int],
        data: bytes,
        clip_id: str | None = None,
        group: str | None = None,
        name: str | None = None,
        source_path: str | None = None,
    ) -> dict[str, Any]:
        dtype = canonical_dtype(dtype)
        shape_list = [int(v) for v in shape]
        byte_length = len(data)
        expected = expected_nbytes(dtype, shape_list)
        if byte_length != expected:
            raise ValueError(
                f"Tensor {logical_path!r} byte length mismatch: got {byte_length}, "
                f"expected {expected} for dtype={dtype} shape={shape_list}"
            )
        spec: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": -1,
            "byteLength": byte_length,
            "dtype": dtype,
            "shape": shape_list,
            "layout": "C",
            "storage": "raw",
            "alignment": self.alignment,
            "path": logical_path,
        }
        if source_path is not None:
            spec["sourcePath"] = source_path
        if name is not None and group is not None:
            spec["semantic"] = tensor_semantic(name, group)
        self.tensors.append(
            TensorPayload(
                logical_path=logical_path,
                dtype=dtype,
                shape=shape_list,
                data=data,
                spec_ref=spec,
                clip_id=clip_id,
                group=group,
                name=name,
            )
        )
        return spec

    def build_payload(self) -> tuple[bytes, list[dict[str, Any]]]:
        chunks: list[bytes] = []
        offset = 0
        index: list[dict[str, Any]] = []
        for i, tensor in enumerate(self.tensors):
            aligned = align_up(offset, self.alignment)
            if aligned > offset:
                chunks.append(b"\x00" * (aligned - offset))
                offset = aligned
            tensor.spec_ref["byteOffset"] = offset
            tensor.spec_ref["byteLength"] = len(tensor.data)
            if self.include_hashes:
                tensor.spec_ref["sha256"] = hashlib.sha256(tensor.data).hexdigest()
            index.append(
                {
                    "index": i,
                    "path": tensor.logical_path,
                    "clipId": tensor.clip_id,
                    "group": tensor.group,
                    "name": tensor.name,
                    "byteOffset": offset,
                    "byteLength": len(tensor.data),
                    "dtype": tensor.dtype,
                    "shape": tensor.shape,
                }
            )
            chunks.append(tensor.data)
            offset += len(tensor.data)
        return b"".join(chunks), index

    def write(
        self,
        *,
        output: Path,
        manifest: dict[str, Any],
        overwrite: bool = False,
        manifest_output: Path | None = None,
    ) -> dict[str, Any]:
        output = Path(output)
        if output.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output}. Pass --overwrite.")
        output.parent.mkdir(parents=True, exist_ok=True)

        payload, tensor_index = self.build_payload()
        payload_length = len(payload)
        manifest = copy.deepcopy(manifest)
        manifest.setdefault("container", {})
        manifest.setdefault("buffers", [])
        manifest["tensorIndex"] = tensor_index

        # Iterate because embedding payloadOffset/manifestLength can change the JSON length.
        payload_offset = align_up(HEADER_BYTES + 2, self.alignment)
        manifest_bytes = b"{}"
        for _ in range(20):
            manifest["container"].update(
                {
                    "magic": MAGIC.decode("ascii"),
                    "formatVersion": RUNTIME_VERSION,
                    "headerBytes": HEADER_BYTES,
                    "byteOrder": "little",
                    "manifestEncoding": "utf-8-json",
                    "manifestOffset": HEADER_BYTES,
                    "manifestLength": len(manifest_bytes),
                    "payloadOffset": payload_offset,
                    "payloadLength": payload_length,
                    "tensorByteOffsets": "relative_to_payload",
                    "alignment": self.alignment,
                }
            )
            manifest["buffers"] = [
                {
                    "index": 0,
                    "byteOffset": payload_offset,
                    "byteLength": payload_length,
                    "alignment": self.alignment,
                    "usage": "tensor-payload",
                    "byteOffsetsAreRelativeTo": "payloadOffset",
                }
            ]
            new_manifest_bytes = json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8")
            new_payload_offset = align_up(HEADER_BYTES + len(new_manifest_bytes), self.alignment)
            if new_payload_offset == payload_offset and len(new_manifest_bytes) == len(manifest_bytes):
                manifest_bytes = new_manifest_bytes
                break
            manifest_bytes = new_manifest_bytes
            payload_offset = new_payload_offset
        else:
            raise RuntimeError("Could not stabilize runtime manifest/header offsets")

        manifest["container"].update(
            {
                "manifestLength": len(manifest_bytes),
                "payloadOffset": payload_offset,
                "payloadLength": payload_length,
            }
        )
        manifest["buffers"][0].update({"byteOffset": payload_offset, "byteLength": payload_length})
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8")
        payload_offset = align_up(HEADER_BYTES + len(manifest_bytes), self.alignment)
        manifest["container"].update(
            {
                "manifestLength": len(manifest_bytes),
                "payloadOffset": payload_offset,
                "payloadLength": payload_length,
            }
        )
        manifest["buffers"][0].update({"byteOffset": payload_offset, "byteLength": payload_length})
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8")
        payload_offset = align_up(HEADER_BYTES + len(manifest_bytes), self.alignment)
        manifest["container"].update(
            {
                "manifestLength": len(manifest_bytes),
                "payloadOffset": payload_offset,
                "payloadLength": payload_length,
            }
        )
        manifest["buffers"][0].update({"byteOffset": payload_offset, "byteLength": payload_length})
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8")
        final_payload_offset = align_up(HEADER_BYTES + len(manifest_bytes), self.alignment)
        if final_payload_offset != payload_offset:
            # One last strict stabilization pass.
            payload_offset = final_payload_offset
            manifest["container"].update({"payloadOffset": payload_offset, "manifestLength": len(manifest_bytes)})
            manifest["buffers"][0]["byteOffset"] = payload_offset
            manifest_bytes = json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8")
            payload_offset = align_up(HEADER_BYTES + len(manifest_bytes), self.alignment)
            manifest["container"].update({"payloadOffset": payload_offset, "manifestLength": len(manifest_bytes)})
            manifest["buffers"][0]["byteOffset"] = payload_offset
            manifest_bytes = json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8")

        header = HEADER_STRUCT.pack(
            MAGIC,
            RUNTIME_VERSION,
            HEADER_BYTES,
            HEADER_BYTES,
            len(manifest_bytes),
            payload_offset,
            payload_length,
            self.alignment,
            0,
        )
        if len(header) != HEADER_BYTES:
            raise AssertionError("unexpected header size")

        tmp = output.with_suffix(output.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        with tmp.open("wb") as f:
            f.write(header)
            f.write(manifest_bytes)
            pad = payload_offset - f.tell()
            if pad < 0:
                raise RuntimeError("manifest overran payload offset")
            f.write(b"\x00" * pad)
            payload_start = f.tell()
            if payload_start != payload_offset:
                raise RuntimeError(f"payload start mismatch: {payload_start} != {payload_offset}")
            f.write(payload)
        tmp.replace(output)

        if manifest_output is not None:
            manifest_output = Path(manifest_output)
            manifest_output.parent.mkdir(parents=True, exist_ok=True)
            manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return manifest


# -----------------------------------------------------------------------------
# Shared timeline helpers
# -----------------------------------------------------------------------------

def parse_range(raw: str, index: int) -> ClipSchedule:
    parts = raw.replace(",", ":").split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"Invalid --range {raw!r}. Expected TRAIN_START:TRAIN_END:CORE_START:CORE_END"
        )
    try:
        train_start, train_end, core_start, core_end = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer in --range {raw!r}") from exc
    sched = ClipSchedule(index, train_start, train_end, core_start, core_end)
    validate_schedule_item(sched)
    return sched


def validate_schedule_item(s: ClipSchedule) -> None:
    if s.train_start < 0 or s.core_start < 0:
        raise ValueError(f"clip_{s.index:04d}: frame starts must be >= 0")
    if s.train_end <= s.train_start:
        raise ValueError(f"clip_{s.index:04d}: train_end must be > train_start")
    if s.core_end <= s.core_start:
        raise ValueError(f"clip_{s.index:04d}: core_end must be > core_start")
    if s.core_start < s.train_start or s.core_end > s.train_end:
        raise ValueError(
            f"clip_{s.index:04d}: core [{s.core_start},{s.core_end}) must be inside "
            f"train [{s.train_start},{s.train_end})"
        )


def _as_int(d: dict[str, Any], key: str, default: int) -> int:
    v = d.get(key, default)
    return int(default if v is None else v)


def normalize_items_from_authoring_manifest(manifest: dict[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Return item_key, clips, and timeline metadata for both timeline and legacy single-clip .trigs."""
    fmt = str(manifest.get("format"))
    if fmt == "trigs.timeline":
        if isinstance(manifest.get("clips"), list):
            return "clips", copy.deepcopy(manifest["clips"]), copy.deepcopy(manifest.get("timeline", {}))
        if isinstance(manifest.get("chunks"), list):
            return "chunks", copy.deepcopy(manifest["chunks"]), copy.deepcopy(manifest.get("timeline", {}))
        raise ValueError("Timeline .trigs manifest contains neither clips nor chunks")

    if fmt == "trigs":
        playback = manifest.get("playback", {})
        fps = float(playback.get("frameRate", 60.0))
        frames = int(playback.get("frames", max(1, round(float(playback.get("durationSeconds", 1.0)) * fps))))
        t_start = float(playback.get("tStart", 0.0))
        t_end = float(playback.get("tEnd", frames / fps))
        item = {
            "id": "clip_0000",
            "index": 0,
            "path": ".",
            "source": {"kind": "legacy-trigs"},
            "global": {
                "frameStart": 0,
                "frameEndExclusive": frames,
                "coreFrameStart": 0,
                "coreFrameEndExclusive": frames,
                "activeFrameStart": 0,
                "activeFrameEndExclusive": frames,
                "timeStart": 0.0,
                "timeEnd": frames / fps,
            },
            "training": {
                "globalFrameStart": 0,
                "globalFrameEndExclusive": frames,
                "localFrameStart": 0,
                "localFrameEndExclusive": frames,
                "frameCount": frames,
            },
            "local": {
                "frameStart": 0,
                "frameEndExclusive": frames,
                "coreFrameStart": 0,
                "coreFrameEndExclusive": frames,
                "timeStart": t_start,
                "timeEnd": t_end,
            },
            "render": {
                "frameCount": frames,
                "fadeInFrames": 0,
                "fadeOutFrames": 0,
                "opacityScale": 1.0,
            },
            "model": {
                "count": manifest.get("count"),
                "shCoeffCount": manifest.get("shCoeffCount"),
                "shDegree": manifest.get("shDegree"),
                "motionModel": "trigs.se3_bezier_v1",
                "tensorLayout": "split_dynamic",
                "staticExportMode": "all-dynamic",
            },
            "dynamic": manifest["dynamic"],
            "static": {"count": 0, "shCoeffCount": manifest.get("shCoeffCount"), "tensors": {}},
        }
        timeline = {
            "frameRate": fps,
            "totalFrames": frames,
            "durationSeconds": frames / fps,
            "timeUnit": "seconds",
            "frameIndexing": "zero_based_end_exclusive",
        }
        return "single", [item], timeline

    raise ValueError(f"Unsupported .trigs manifest format: {fmt!r}")


def default_runtime_manifest_base(
    *,
    source_kind: str,
    source_path: str | None,
    timeline: dict[str, Any],
    clips: list[dict[str, Any]],
    packaging: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    playback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": RUNTIME_FORMAT,
        "version": RUNTIME_VERSION,
        "extension": ".trigsbin",
        "createdBy": Path(__file__).name,
        "createdByVersion": VERSION,
        "createdAtUnix": int(time.time()),
        "source": {"kind": source_kind, "path": source_path},
        "semantics": {
            "timelineIsStandalone": True,
            "clipsAreInternalOnly": True,
            "timeUnit": "seconds",
            "frameIndexing": "zero_based_end_exclusive",
            "localFrameZeroIsTrainingGlobalFrameStart": True,
            "globalCoreRangeMeansFullOpacityRange": True,
            "tensorStorage": "raw-aligned-SoA",
            "tensorByteOffsets": "relative_to_payload",
            "quaternionOrder": "xyzw",
            "shChannelOrder": "rgb",
            "scaleValueDomain": "log",
            "opacityValueDomain": "logit",
            "dynamicMotionModel": "trigs.se3_bezier_v1",
            "engineNeutral": True,
        },
        "runtimeHints": {
            "static": {
                "gpuResidency": "load-once-per-active-clip",
                "mutability": "immutable-during-playback",
            },
            "dynamic": {
                "gpuResidency": "load-motion-params-per-active-clip",
                "evaluation": "per-frame-output-buffer",
            },
            "recommendedRenderBatching": "persistent-buffer-regions-static-plus-dynamic",
            "recommendedUpload": "mmap-or-staging-copy-to-gpu-buffer",
        },
        "tensorSchema": {
            "groups": {
                "static": list(STATIC_KEYS),
                "dynamic": list(REQUIRED_KEYS),
                "flags": list(OPTIONAL_FLAG_KEYS),
            },
            "dtypeEndianness": "little",
            "arrayLayout": "C-contiguous",
        },
        "metadata": metadata or {},
        "timeline": timeline,
        "packaging": packaging or {},
        "playback": playback or {},
        "clips": clips,
    }


# -----------------------------------------------------------------------------
# from-trigs path
# -----------------------------------------------------------------------------

def read_json_from_zip(zf: zipfile.ZipFile, path: str) -> dict[str, Any]:
    return json.loads(zf.read(path).decode("utf-8"))


def copy_tensor_group_from_zip(
    *,
    zf: zipfile.ZipFile,
    builder: RuntimeBinaryBuilder,
    clip_id: str,
    group_name: str,
    group: dict[str, Any] | None,
) -> dict[str, Any]:
    if not group:
        return {"count": 0, "shCoeffCount": None, "tensors": {}}
    out = copy.deepcopy(group)
    tensors = group.get("tensors", {})
    out_tensors: dict[str, dict[str, Any]] = {}
    for name, spec in tensors.items():
        if not isinstance(spec, dict):
            raise TypeError(f"{clip_id}/{group_name}/{name}: tensor spec must be a dict")
        source_path = str(spec.get("path", ""))
        if not source_path:
            raise ValueError(f"{clip_id}/{group_name}/{name}: missing tensor path")
        dtype = canonical_dtype(str(spec["dtype"]))
        shape = [int(v) for v in spec["shape"]]
        data = zf.read(source_path)
        if len(data) != expected_nbytes(dtype, shape):
            raise ValueError(
                f"{source_path}: byte length mismatch: got {len(data)}, "
                f"expected {expected_nbytes(dtype, shape)}"
            )
        logical_path = f"clips/{clip_id}/{group_name}/{name}.{dtype_ext(dtype)}"
        out_tensors[name] = builder.add_tensor(
            logical_path=logical_path,
            dtype=dtype,
            shape=shape,
            data=data,
            clip_id=clip_id,
            group=group_name,
            name=name,
            source_path=source_path,
        )
    out["tensors"] = out_tensors
    return out


def convert_from_trigs(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if not src.is_file():
        raise SystemExit(f"Input .trigs not found: {src}")
    builder = RuntimeBinaryBuilder(alignment=args.alignment, include_hashes=args.tensor_hashes)

    with zipfile.ZipFile(src, mode="r") as zf:
        manifest = read_json_from_zip(zf, MANIFEST_NAME)
        item_key, items, timeline = normalize_items_from_authoring_manifest(manifest)

        runtime_clips: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            clip = copy.deepcopy(item)
            clip_id = str(clip.get("id", f"clip_{idx:04d}"))
            clip["id"] = clip_id
            clip.setdefault("index", idx)
            clip["dynamic"] = copy_tensor_group_from_zip(
                zf=zf,
                builder=builder,
                clip_id=clip_id,
                group_name="dynamic",
                group=item.get("dynamic"),
            )
            clip["static"] = copy_tensor_group_from_zip(
                zf=zf,
                builder=builder,
                clip_id=clip_id,
                group_name="static",
                group=item.get("static"),
            )
            if "flags" in item:
                clip["flags"] = copy_tensor_group_from_zip(
                    zf=zf,
                    builder=builder,
                    clip_id=clip_id,
                    group_name="flags",
                    group=item.get("flags"),
                )
            runtime_clips.append(clip)

    packaging = copy.deepcopy(manifest.get("packaging", {}))
    packaging.update(
        {
            "runtimePackagerVersion": VERSION,
            "sourceAuthoringFormat": manifest.get("format"),
            "sourceManifestItemKey": item_key,
            "payloadEncoding": "single-file-raw-aligned",
            "alignment": int(args.alignment),
        }
    )
    metadata = copy.deepcopy(manifest.get("metadata", {}))
    metadata.setdefault("sourceFileName", src.name)
    playback = copy.deepcopy(manifest.get("playback", {}))

    runtime_manifest = default_runtime_manifest_base(
        source_kind="trigs-authoring-archive",
        source_path=os.fspath(src),
        timeline=timeline,
        clips=runtime_clips,
        packaging=packaging,
        metadata=metadata,
        playback=playback,
    )
    written = builder.write(
        output=Path(args.output),
        manifest=runtime_manifest,
        overwrite=args.overwrite,
        manifest_output=args.manifest_output,
    )
    print_summary(Path(args.output), written)
    return 0


# -----------------------------------------------------------------------------
# from-checkpoints path
# -----------------------------------------------------------------------------

def import_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("from-checkpoints requires PyTorch to read trainer checkpoint .pt files") from exc
    return torch


def read_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    out = list(args.checkpoint)
    if args.checkpoint_list is not None:
        with Path(args.checkpoint_list).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                out.append(Path(line))
    return out


def to_float32_numpy(value: Any, name: str) -> np.ndarray:
    torch = import_torch()
    if isinstance(value, torch.nn.Parameter):
        value = value.detach()
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError(f'Tensor "{name}" must be strided, got {value.layout}')
        value = value.detach().cpu().contiguous().to(torch.float32).numpy()
    elif isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value, dtype=np.float32)
    else:
        raise TypeError(f'Unsupported value for tensor "{name}": {type(value).__name__}')
    return np.asarray(value, dtype=np.float32, order="C")


def to_bool_numpy(value: Any, name: str, count: int) -> np.ndarray:
    torch = import_torch()
    if isinstance(value, torch.nn.Parameter):
        value = value.detach()
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError(f'Tensor "{name}" must be strided, got {value.layout}')
        arr = value.detach().cpu().contiguous().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    else:
        raise TypeError(f'Unsupported value for flag "{name}": {type(value).__name__}')

    arr = np.asarray(arr).reshape(-1)
    if len(arr) != int(count):
        raise ValueError(f'Flag "{name}" expected {count} rows, got {len(arr)}')
    return np.ascontiguousarray(arr > 0.5, dtype=bool)


def normalize_tensor_shape(name: str, array: np.ndarray) -> np.ndarray:
    if name == "sh0" and array.ndim == 2 and array.shape[1] == 3:
        return np.ascontiguousarray(array[:, None, :], dtype=np.float32)
    if name in ("opacities", "times", "log_durations") and array.ndim == 2 and array.shape[1] == 1:
        return np.ascontiguousarray(array.reshape(array.shape[0]), dtype=np.float32)
    return array


def expect_shape(name: str, array: np.ndarray, ndim: int, trailing: tuple[int, ...]) -> None:
    if array.ndim != ndim:
        raise ValueError(f'Tensor "{name}" must be rank {ndim}, got shape {list(array.shape)}')
    if tuple(array.shape[-len(trailing):]) != trailing:
        raise ValueError(f'Tensor "{name}" must end with shape {list(trailing)}, got {list(array.shape)}')


def validate_shape(name: str, array: np.ndarray) -> None:
    if name in (
        "means", "scales", "anchors", "zeta_base_r", "zeta_base_t",
        "zeta_c0_r", "zeta_c0_t", "zeta_c1_r", "zeta_c1_t", "zeta_c2_r", "zeta_c2_t",
    ):
        expect_shape(name, array, ndim=2, trailing=(3,))
    elif name == "quats":
        expect_shape(name, array, ndim=2, trailing=(4,))
    elif name in ("opacities", "times", "log_durations"):
        if array.ndim != 1:
            raise ValueError(f'Tensor "{name}" must be shape [N] after normalization, got {list(array.shape)}')
    elif name == "sh0":
        expect_shape(name, array, ndim=3, trailing=(1, 3))
    elif name == "shN":
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f'Tensor "{name}" must be shape [N,K,3], got {list(array.shape)}')


def dtype_for_tensor(name: str, policy: str) -> str:
    if policy == "float32":
        return "float32"
    if policy == "float16":
        return "float16"
    if policy == "mixed":
        return "float32" if name in TEMPORAL_FLOAT32_KEYS else "float16"
    if policy == "motion32":
        return "float32" if name in MOTION_FLOAT32_KEYS else "float16"
    raise ValueError(f"Unsupported tensor dtype policy: {policy}")


def load_checkpoint_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    torch = import_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"{path}: expected checkpoint dict, got {type(ckpt).__name__}")
    params = ckpt.get("params")
    if not isinstance(params, dict):
        raise KeyError(f'{path}: missing checkpoint["params"]')

    arrays: dict[str, np.ndarray] = {}
    flags: dict[str, np.ndarray] = {}
    count: int | None = None

    for key in REQUIRED_KEYS:
        value = params.get(key)
        if value is None:
            if key == "shN" and count is not None:
                arr = np.zeros((count, 0, 3), dtype=np.float32)
            else:
                raise KeyError(f'{path}: missing required tensor checkpoint["params"]["{key}"]')
        else:
            arr = to_float32_numpy(value, key)
        arr = normalize_tensor_shape(key, arr)
        validate_shape(key, arr)
        if key == "means":
            count = int(arr.shape[0])
        elif count is None:
            raise RuntimeError("means must be loaded before other tensors")
        elif int(arr.shape[0]) != count:
            raise ValueError(f'{path}: leading dimension mismatch for "{key}": expected {count}, got {arr.shape[0]}')
        arrays[key] = arr

    if count is None:
        count = 0

    for key in OPTIONAL_FLAG_KEYS:
        value = params.get(key)
        if value is None:
            flags[key] = np.zeros((count,), dtype=bool)
        else:
            flags[key] = to_bool_numpy(value, key, count)

    sh_coeff_count = int(arrays["shN"].shape[1])
    static_count = int(flags["is_static"].sum())
    dynamic_protect_count = int(flags["dynamic_protect"].sum())
    meta: dict[str, Any] = {
        "count": int(count or 0),
        "staticCount": static_count,
        "dynamicCount": int(count - static_count),
        "dynamicProtectCount": dynamic_protect_count,
        "shCoeffCount": sh_coeff_count,
        "shDegree": max(0, int(math.isqrt(sh_coeff_count + 1) - 1)),
        "step": int(ckpt["step"]) if isinstance(ckpt.get("step"), (int, np.integer)) else None,
    }
    for k in ("fps", "n_frames", "selfcap_eval_meta"):
        if k in ckpt:
            try:
                json.dumps(ckpt[k])
                meta[k] = ckpt[k]
            except TypeError:
                pass

    # Chunked trainer checkpoints retain the exact temporal windows and the
    # midpoint ownership boundaries used during training.  Keep just the
    # small, JSON-safe portion here so a single assembled checkpoint can be
    # partitioned back into runtime chunks without importing trainer code.
    metrics = ckpt.get("metrics")
    if isinstance(metrics, dict):
        raw_windows = metrics.get("windows")
        if isinstance(raw_windows, (list, tuple)):
            try:
                meta["trainerWindows"] = [
                    [float(window[0]), float(window[1])]
                    for window in raw_windows
                    if isinstance(window, (list, tuple)) and len(window) == 2
                ]
            except (TypeError, ValueError):
                meta.pop("trainerWindows", None)

        raw_bounds = metrics.get("ownership_bounds")
        if isinstance(raw_bounds, (list, tuple)):
            try:
                meta["trainerOwnershipBounds"] = [float(value) for value in raw_bounds]
            except (TypeError, ValueError):
                meta.pop("trainerOwnershipBounds", None)

        diagnostics = metrics.get("entry_diagnostics")
        if isinstance(diagnostics, (list, tuple)):
            for diagnostic in reversed(diagnostics):
                if not isinstance(diagnostic, dict) or "cull_threshold" not in diagnostic:
                    continue
                try:
                    meta["trainerCullThreshold"] = float(diagnostic["cull_threshold"])
                except (TypeError, ValueError):
                    pass
                break

    # Training snapshots, and the export copy written by
    # prepare_export_checkpoint.py, carry the run's training config.
    config_threshold = config_temporal_cull_threshold(ckpt.get("config"))
    if config_threshold is not None:
        meta["configCullThreshold"] = config_threshold
    return arrays, flags, meta


def config_temporal_cull_threshold(config: Any) -> float | None:
    """The trainer's effective temporal cull threshold from a saved training config.

    Mirrors ``TrigsTrainer._effective_temporal_cull_threshold``: the
    ``temporal_active_cull_threshold`` override when > 0, never above
    ``temporal_active_threshold``.
    """
    if config is not None and not isinstance(config, dict):
        config = getattr(config, "__dict__", None)
    if not isinstance(config, dict) or "temporal_active_threshold" not in config:
        return None
    try:
        base = float(config["temporal_active_threshold"])
        override = float(config.get("temporal_active_cull_threshold") or 0.0)
    except (TypeError, ValueError):
        return None
    value = override if override > 0.0 else base
    return min(max(value, 1e-8), min(max(base, 1e-8), 1.0 - 1e-8))


def _slice_arrays(arrays: dict[str, np.ndarray], keys: tuple[str, ...], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.ascontiguousarray(arrays[key][mask]) for key in keys}


def add_array_group_to_builder(
    *,
    builder: RuntimeBinaryBuilder,
    arrays: dict[str, np.ndarray],
    clip_id: str,
    group_name: str,
    dtype_policy: str,
) -> dict[str, dict[str, Any]]:
    tensors: dict[str, dict[str, Any]] = {}
    for name, arr in arrays.items():
        dtype = dtype_for_tensor(name, dtype_policy)
        logical_path = f"clips/{clip_id}/{group_name}/{name}.{dtype_ext(dtype)}"
        tensors[name] = builder.add_tensor(
            logical_path=logical_path,
            dtype=dtype,
            shape=[int(v) for v in arr.shape],
            data=encode_array(arr, dtype),
            clip_id=clip_id,
            group=group_name,
            name=name,
        )
    return tensors


def add_global_static_group_to_builder(
    *,
    builder: RuntimeBinaryBuilder,
    arrays: dict[str, np.ndarray],
    dtype_policy: str,
    static_id: str = "static_0000",
) -> dict[str, dict[str, Any]]:
    """Write one shot-global static tensor group.

    The tensor payload is stored once at top-level paths such as
    ``static/static_0000/means.f32``. Dynamic clips reference this group by
    ``sourceStaticId`` instead of owning or duplicating static tensors.
    """
    tensors: dict[str, dict[str, Any]] = {}
    for name, arr in arrays.items():
        dtype = dtype_for_tensor(name, dtype_policy)
        logical_path = f"static/{static_id}/{name}.{dtype_ext(dtype)}"
        tensors[name] = builder.add_tensor(
            logical_path=logical_path,
            dtype=dtype,
            shape=[int(v) for v in arr.shape],
            data=encode_array(arr, dtype),
            clip_id=static_id,
            group="static",
            name=name,
        )
    return tensors


def extract_static_arrays_from_checkpoint_payload(
    *,
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
    flags: dict[str, np.ndarray],
    meta: dict[str, Any],
    static_export_mode: str,
) -> tuple[dict[str, np.ndarray], int, int, int]:
    """Return static arrays/counts according to the same split policy used by clips."""
    count = int(meta["count"])
    sh_coeff_count = int(meta["shCoeffCount"])
    sh_degree = int(meta["shDegree"])

    if static_export_mode == "split":
        static_mask = np.asarray(flags.get("is_static", np.zeros((count,), dtype=bool)), dtype=bool).reshape(-1)
        if len(static_mask) != count:
            raise ValueError(f"{checkpoint_path}: is_static mask length mismatch: {len(static_mask)} != {count}")
    elif static_export_mode == "all-dynamic":
        static_mask = np.zeros((count,), dtype=bool)
    else:
        raise ValueError(f"Unsupported static export mode: {static_export_mode}")

    static_arrays = _slice_arrays(arrays, STATIC_KEYS, static_mask)
    return static_arrays, int(static_mask.sum()), sh_coeff_count, sh_degree




def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def _robust_sh0_anchor(
    arrays: dict[str, np.ndarray],
    flags: dict[str, np.ndarray],
    *,
    mode: str,
    min_samples: int,
) -> tuple[np.ndarray, int, str]:
    """Return a robust per-channel SH0 anchor for optional clip color matching.

    This is intentionally conservative. SH coefficients are not literal sRGB,
    but sh0 is the DC/base color term in this packager's semantic model. We use
    a trimmed median of reasonably opaque rows to estimate clip-level brightness.
    """
    sh0 = np.asarray(arrays["sh0"], dtype=np.float64)
    if sh0.ndim != 3 or sh0.shape[1] != 1 or sh0.shape[2] != 3:
        raise ValueError(f"sh0 expected shape [N,1,3], got {list(sh0.shape)}")

    n = int(sh0.shape[0])
    values = sh0[:, 0, :]
    finite = np.all(np.isfinite(values), axis=1)

    is_static = np.asarray(flags.get("is_static", np.zeros((n,), dtype=bool)), dtype=bool).reshape(-1)
    dynamic_protect = np.asarray(flags.get("dynamic_protect", np.zeros((n,), dtype=bool)), dtype=bool).reshape(-1)

    if len(is_static) != n:
        is_static = np.zeros((n,), dtype=bool)
    if len(dynamic_protect) != n:
        dynamic_protect = np.zeros((n,), dtype=bool)

    if mode == "dynamic-sh0-median":
        mask = finite & (~is_static)
        used_mode = "dynamic"
    elif mode == "dynamic-protect-sh0-median":
        mask = finite & dynamic_protect
        used_mode = "dynamic_protect"
    elif mode == "all-sh0-median":
        mask = finite
        used_mode = "all"
    else:
        raise ValueError(f"Unsupported color normalization mode: {mode}")

    # Avoid mostly transparent rows dominating the estimate.
    if "opacities" in arrays:
        try:
            alpha = _sigmoid_np(np.asarray(arrays["opacities"], dtype=np.float64).reshape(-1))
            if len(alpha) == n and int(mask.sum()) >= max(10, int(min_samples)):
                alpha_sel = alpha[mask]
                thresh = float(np.percentile(alpha_sel, 25.0))
                mask = mask & (alpha >= thresh)
        except Exception:
            pass

    if int(mask.sum()) < int(min_samples):
        # Fallback to all finite rows if the preferred subset is too small.
        fallback = finite
        if "opacities" in arrays:
            try:
                alpha = _sigmoid_np(np.asarray(arrays["opacities"], dtype=np.float64).reshape(-1))
                if len(alpha) == n and int(fallback.sum()) >= max(10, int(min_samples)):
                    thresh = float(np.percentile(alpha[fallback], 25.0))
                    fallback = fallback & (alpha >= thresh)
            except Exception:
                pass
        mask = fallback
        used_mode = f"{used_mode}->all_fallback"

    vals = values[mask]
    if len(vals) == 0:
        return np.ones((3,), dtype=np.float64), 0, used_mode

    lo = np.percentile(vals, 10.0, axis=0)
    hi = np.percentile(vals, 90.0, axis=0)
    vals = np.clip(vals, lo[None, :], hi[None, :])
    anchor = np.median(vals, axis=0).astype(np.float64)

    # If a channel's median is too close to zero, fall back to signed mean of
    # the trimmed values; if still bad, make that channel neutral.
    mean = np.mean(vals, axis=0).astype(np.float64)
    bad = ~np.isfinite(anchor) | (np.abs(anchor) < 1e-8)
    anchor[bad] = mean[bad]
    bad = ~np.isfinite(anchor) | (np.abs(anchor) < 1e-8)
    anchor[bad] = 1.0

    return anchor, int(len(vals)), used_mode


def apply_clip_color_normalization(
    arrays: dict[str, np.ndarray],
    flags: dict[str, np.ndarray],
    *,
    mode: str,
    reference_anchor: np.ndarray | None,
    max_gain: float,
    strength: float,
    apply_to: str,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str]:
    """Optionally scale SH color coefficients for clip-level continuity.

    Returns: (current_anchor_before, reference_anchor_used, gain, sample_count, used_mode)
    """
    current_anchor, sample_count, used_mode = _robust_sh0_anchor(
        arrays,
        flags,
        mode=mode,
        min_samples=min_samples,
    )

    if reference_anchor is None:
        return current_anchor, current_anchor.copy(), np.ones((3,), dtype=np.float64), sample_count, used_mode

    ref = np.asarray(reference_anchor, dtype=np.float64).reshape(3)
    cur = np.asarray(current_anchor, dtype=np.float64).reshape(3)

    raw_gain = np.ones((3,), dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(cur) & (np.abs(cur) > 1e-8)
    raw_gain[valid] = ref[valid] / cur[valid]

    # Negative gains are not useful for color continuity; leave those channels alone.
    raw_gain[~np.isfinite(raw_gain) | (raw_gain <= 0.0)] = 1.0

    strength = float(np.clip(strength, 0.0, 1.0))
    gain = 1.0 + (raw_gain - 1.0) * strength

    max_gain = max(float(max_gain), 1.0)
    gain = np.clip(gain, 1.0 / max_gain, max_gain)

    arrays["sh0"] = np.ascontiguousarray(
        np.asarray(arrays["sh0"], dtype=np.float32) * gain.reshape(1, 1, 3).astype(np.float32)
    )

    if apply_to == "sh0-shN" and "shN" in arrays and np.asarray(arrays["shN"]).size > 0:
        arrays["shN"] = np.ascontiguousarray(
            np.asarray(arrays["shN"], dtype=np.float32) * gain.reshape(1, 1, 3).astype(np.float32)
        )

    return current_anchor, ref, gain, sample_count, used_mode



def _row_global_frame_centers(
    arrays: dict[str, np.ndarray],
    schedule: ClipSchedule,
    frame_rate: float,
) -> np.ndarray:
    """Approximate each primitive's global frame center from local training time.

    Trainer checkpoints store params["times"] in seconds local to the training
    window. Timeline schedules are zero-based/end-exclusive global frames, so a
    primitive centered at local time t maps to:

        global_frame ~= schedule.train_start + t * frame_rate
    """
    t = np.asarray(arrays.get("times"), dtype=np.float64).reshape(-1)
    return float(schedule.train_start) + t * float(frame_rate)


def _mask_for_overlap_anchor(
    arrays: dict[str, np.ndarray],
    flags: dict[str, np.ndarray],
    schedule: ClipSchedule,
    *,
    frame_rate: float,
    overlap_start: int,
    overlap_end: int,
    mode: str,
    min_alpha_percentile: float,
) -> tuple[np.ndarray, str]:
    n = int(np.asarray(arrays["sh0"]).shape[0])
    finite = np.ones((n,), dtype=bool)
    for key in ("sh0", "times"):
        arr = np.asarray(arrays[key])
        finite &= np.all(np.isfinite(arr.reshape(n, -1)), axis=1)

    frame_centers = _row_global_frame_centers(arrays, schedule, frame_rate)
    # Half-open overlap, with a small tolerance because times can be float.
    temporal = (
        (frame_centers >= float(overlap_start) - 0.51)
        & (frame_centers < float(overlap_end) + 0.51)
    )

    is_static = np.asarray(flags.get("is_static", np.zeros((n,), dtype=bool)), dtype=bool).reshape(-1)
    dynamic_protect = np.asarray(flags.get("dynamic_protect", np.zeros((n,), dtype=bool)), dtype=bool).reshape(-1)
    if len(is_static) != n:
        is_static = np.zeros((n,), dtype=bool)
    if len(dynamic_protect) != n:
        dynamic_protect = np.zeros((n,), dtype=bool)

    if mode.startswith("dynamic-protect-"):
        mask = finite & temporal & dynamic_protect
        used = "dynamic_protect_overlap"
    elif mode.startswith("dynamic-"):
        mask = finite & temporal & (~is_static)
        used = "dynamic_overlap"
    elif mode.startswith("all-"):
        mask = finite & temporal
        used = "all_overlap"
    else:
        raise ValueError(f"Unsupported overlap color calibration mode: {mode}")

    # Prefer rows with non-trivial opacity so dead/transparent primitives do not dominate.
    if "opacities" in arrays and int(mask.sum()) > 0:
        try:
            alpha = _sigmoid_np(np.asarray(arrays["opacities"], dtype=np.float64).reshape(-1))
            if len(alpha) == n:
                pct = float(np.clip(min_alpha_percentile, 0.0, 95.0))
                thresh = float(np.percentile(alpha[mask], pct))
                mask = mask & (alpha >= thresh)
                used += f"_alpha_p{pct:g}"
        except Exception:
            pass

    return mask, used


def _trimmed_median_rgb(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) == 0:
        return np.ones((3,), dtype=np.float64)
    lo = np.percentile(values, 10.0, axis=0)
    hi = np.percentile(values, 90.0, axis=0)
    values = np.clip(values, lo[None, :], hi[None, :])
    anchor = np.median(values, axis=0).astype(np.float64)
    mean = np.mean(values, axis=0).astype(np.float64)
    bad = ~np.isfinite(anchor) | (np.abs(anchor) < 1e-8)
    anchor[bad] = mean[bad]
    bad = ~np.isfinite(anchor) | (np.abs(anchor) < 1e-8)
    anchor[bad] = 1.0
    return anchor


def _overlap_color_anchor(
    arrays: dict[str, np.ndarray],
    flags: dict[str, np.ndarray],
    schedule: ClipSchedule,
    *,
    frame_rate: float,
    overlap_start: int,
    overlap_end: int,
    mode: str,
    min_samples: int,
    min_alpha_percentile: float,
) -> tuple[np.ndarray, int, str]:
    mask, used = _mask_for_overlap_anchor(
        arrays,
        flags,
        schedule,
        frame_rate=frame_rate,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        mode=mode,
        min_alpha_percentile=min_alpha_percentile,
    )

    n = int(np.asarray(arrays["sh0"]).shape[0])
    is_static = np.asarray(flags.get("is_static", np.zeros((n,), dtype=bool)), dtype=bool).reshape(-1)
    dynamic_protect = np.asarray(flags.get("dynamic_protect", np.zeros((n,), dtype=bool)), dtype=bool).reshape(-1)
    if len(is_static) != n:
        is_static = np.zeros((n,), dtype=bool)
    if len(dynamic_protect) != n:
        dynamic_protect = np.zeros((n,), dtype=bool)

    # If overlap subset is too thin, fall back to the same semantic group over
    # the whole clip. This keeps the test from becoming unstable with sparse
    # keyframe-step=2 rows in a short overlap.
    if int(mask.sum()) < int(min_samples):
        if mode.startswith("dynamic-protect-"):
            fallback = dynamic_protect
            used += "->whole_clip_dynamic_protect_fallback"
        elif mode.startswith("dynamic-"):
            fallback = ~is_static
            used += "->whole_clip_dynamic_fallback"
        elif mode.startswith("all-"):
            fallback = np.ones((n,), dtype=bool)
            used += "->whole_clip_all_fallback"
        else:
            fallback = np.ones((n,), dtype=bool)
        finite = np.all(np.isfinite(np.asarray(arrays["sh0"]).reshape(n, -1)), axis=1)
        mask = fallback & finite

    sample_count = int(mask.sum())
    if sample_count <= 0:
        return np.ones((3,), dtype=np.float64), 0, used

    if mode.endswith("shN-rms"):
        shn = np.asarray(arrays.get("shN"), dtype=np.float64)
        if shn.ndim != 3 or shn.shape[1] == 0:
            # No higher-order SH present; fall back to sh0.
            sh0 = np.asarray(arrays["sh0"], dtype=np.float64)[:, 0, :]
            return _trimmed_median_rgb(sh0[mask]), sample_count, used + "->sh0_no_shN"
        # Per-row, per-channel higher-order SH energy. This is positive, so a
        # gain ratio is meaningful even though signed shN medians are not.
        vals = np.sqrt(np.mean(np.square(shn[mask]), axis=1))
        return _trimmed_median_rgb(vals), sample_count, used + "_shN_rms"

    # Default: sh0 median anchor.
    sh0 = np.asarray(arrays["sh0"], dtype=np.float64)[:, 0, :]
    return _trimmed_median_rgb(sh0[mask]), sample_count, used + "_sh0_median"


def _gain_from_anchors(
    reference_anchor: np.ndarray,
    current_anchor: np.ndarray,
    *,
    max_gain: float,
    strength: float,
) -> np.ndarray:
    ref = np.asarray(reference_anchor, dtype=np.float64).reshape(3)
    cur = np.asarray(current_anchor, dtype=np.float64).reshape(3)
    raw = np.ones((3,), dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(cur) & (np.abs(cur) > 1e-8)
    raw[valid] = ref[valid] / cur[valid]
    raw[~np.isfinite(raw) | (raw <= 0.0)] = 1.0
    strength = float(np.clip(strength, 0.0, 1.0))
    gain = 1.0 + (raw - 1.0) * strength
    max_gain = max(float(max_gain), 1.0)
    return np.clip(gain, 1.0 / max_gain, max_gain)


def _dynamic_target_mask_for_gain(
    arrays: dict[str, np.ndarray],
    flags: dict[str, np.ndarray],
) -> np.ndarray:
    n = int(np.asarray(arrays["sh0"]).shape[0])
    is_static = np.asarray(flags.get("is_static", np.zeros((n,), dtype=bool)), dtype=bool).reshape(-1)
    if len(is_static) != n:
        is_static = np.zeros((n,), dtype=bool)
    return ~is_static


def _apply_rgb_gain_to_sh(
    arrays: dict[str, np.ndarray],
    *,
    gain: np.ndarray,
    target_mask: np.ndarray,
    apply_to: str,
) -> None:
    gain32 = np.asarray(gain, dtype=np.float32).reshape(1, 1, 3)
    target_mask = np.asarray(target_mask, dtype=bool).reshape(-1)
    if len(target_mask) != int(np.asarray(arrays["sh0"]).shape[0]):
        raise ValueError("target mask length mismatch for color gain")

    sh0 = np.asarray(arrays["sh0"], dtype=np.float32).copy()
    sh0[target_mask] = sh0[target_mask] * gain32
    arrays["sh0"] = np.ascontiguousarray(sh0)

    if apply_to == "sh0-shN" and "shN" in arrays and np.asarray(arrays["shN"]).size > 0:
        shn = np.asarray(arrays["shN"], dtype=np.float32).copy()
        shn[target_mask] = shn[target_mask] * gain32
        arrays["shN"] = np.ascontiguousarray(shn)


def apply_overlap_color_calibration(
    previous_arrays: dict[str, np.ndarray],
    previous_flags: dict[str, np.ndarray],
    previous_schedule: ClipSchedule,
    arrays: dict[str, np.ndarray],
    flags: dict[str, np.ndarray],
    schedule: ClipSchedule,
    *,
    frame_rate: float,
    mode: str,
    apply_to: str,
    max_gain: float,
    strength: float,
    min_samples: int,
    min_alpha_percentile: float,
) -> dict[str, Any]:
    overlap_start = max(int(previous_schedule.train_start), int(schedule.train_start))
    overlap_end = min(int(previous_schedule.train_end), int(schedule.train_end))

    if overlap_end <= overlap_start:
        return {
            "enabled": True,
            "skipped": True,
            "reason": "no_train_overlap",
            "overlapStart": int(overlap_start),
            "overlapEnd": int(overlap_end),
            "gain": [1.0, 1.0, 1.0],
        }

    ref_anchor, ref_samples, ref_mode = _overlap_color_anchor(
        previous_arrays,
        previous_flags,
        previous_schedule,
        frame_rate=frame_rate,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        mode=mode,
        min_samples=min_samples,
        min_alpha_percentile=min_alpha_percentile,
    )
    cur_anchor, cur_samples, cur_mode = _overlap_color_anchor(
        arrays,
        flags,
        schedule,
        frame_rate=frame_rate,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        mode=mode,
        min_samples=min_samples,
        min_alpha_percentile=min_alpha_percentile,
    )

    gain = _gain_from_anchors(
        ref_anchor,
        cur_anchor,
        max_gain=max_gain,
        strength=strength,
    )
    target_mask = _dynamic_target_mask_for_gain(arrays, flags)
    _apply_rgb_gain_to_sh(arrays, gain=gain, target_mask=target_mask, apply_to=apply_to)

    return {
        "enabled": True,
        "skipped": False,
        "mode": mode,
        "applyTo": apply_to,
        "overlapStart": int(overlap_start),
        "overlapEnd": int(overlap_end),
        "referenceSamples": int(ref_samples),
        "currentSamples": int(cur_samples),
        "referenceUsedMode": ref_mode,
        "currentUsedMode": cur_mode,
        "referenceAnchor": [float(v) for v in ref_anchor],
        "currentAnchor": [float(v) for v in cur_anchor],
        "gain": [float(v) for v in gain],
        "dynamicRowsScaled": int(np.asarray(target_mask, dtype=bool).sum()),
    }

def static_group_signature(static_arrays: dict[str, np.ndarray], dtype_policy: str) -> str:
    """Hash the static tensor payload exactly as it would be encoded."""
    h = hashlib.sha256()
    for name in STATIC_KEYS:
        arr = np.ascontiguousarray(static_arrays[name])
        dtype = dtype_for_tensor(name, dtype_policy)
        shape = [int(v) for v in arr.shape]
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(dtype.encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(shape, separators=(",", ":")).encode("utf-8"))
        h.update(b"\0")
        h.update(encode_array(arr, dtype))
        h.update(b"\0")
    return h.hexdigest()


def static_tensor_specs_compatible(
    reference_tensors: dict[str, dict[str, Any]],
    static_arrays: dict[str, np.ndarray],
    dtype_policy: str,
) -> bool:
    for name in STATIC_KEYS:
        if name not in reference_tensors or name not in static_arrays:
            return False
        ref = reference_tensors[name]
        dtype = dtype_for_tensor(name, dtype_policy)
        shape = [int(v) for v in np.asarray(static_arrays[name]).shape]
        if str(ref.get("dtype")) != dtype:
            return False
        if [int(v) for v in ref.get("shape", [])] != shape:
            return False
    return True


# TrigsTrainingConfig.temporal_active_threshold default (open4dgs-trigs and its fork).
TRAINER_DEFAULT_TEMPORAL_THRESHOLD = 0.01


def resolve_temporal_threshold(
    cli_value: float | None, meta: dict[str, Any]
) -> tuple[float, str]:
    """Temporal opacity threshold for row selection and the manifest, with its source.

    Order: --temporal-threshold, the trainer's chunk metric, the checkpoint's saved
    training config, then the trainer default. The packager used to fall back to
    0.05, which does not match a run trained with the default 0.01.
    """
    if cli_value is not None:
        return float(cli_value), "command_line"
    if "trainerCullThreshold" in meta:
        return float(meta["trainerCullThreshold"]), "trainer_metrics"
    if "configCullThreshold" in meta:
        return float(meta["configCullThreshold"]), "checkpoint_config"
    return TRAINER_DEFAULT_TEMPORAL_THRESHOLD, "trainer_default"


def balanced_chunk_duration(t_max: float, max_chunk_duration_s: float) -> float:
    """Chunk length that splits [0, t_max] into equal chunks of at most the maximum.

    A sequence no longer than the maximum stays one clip (the returned length is
    >= t_max). Equal chunks avoid a short tail clip that stores the rows crossing
    its boundary for little playback time.
    """
    max_chunk_duration_s = float(max_chunk_duration_s)
    if max_chunk_duration_s <= 0.0:
        raise ValueError("--max-chunk-duration-seconds must be > 0")
    if t_max <= max_chunk_duration_s:
        return max(max_chunk_duration_s, float(t_max))
    n_chunks = math.ceil(float(t_max) / max_chunk_duration_s - 1e-9)
    return float(t_max) / n_chunks


def temporal_radius_scale(opacity_threshold: float) -> float:
    """Match the trainer's finite temporal support radius ``k * sigma``."""
    threshold = min(max(float(opacity_threshold), 1e-8), 1.0 - 1e-8)
    return math.sqrt(-2.0 * math.log(threshold))


def temporal_support_mask(
    arrays: dict[str, np.ndarray],
    flags: dict[str, np.ndarray],
    *,
    t_start: float,
    t_end: float,
    opacity_threshold: float,
    extra_margin_s: float = 0.0,
    include_static: bool = False,
) -> np.ndarray:
    """Select rows whose finite temporal support intersects a time window."""
    times = np.asarray(arrays["times"], dtype=np.float64).reshape(-1)
    sigma = np.exp(
        np.asarray(arrays["log_durations"], dtype=np.float64).reshape(-1)
    )
    sigma = np.maximum(sigma, 1e-6)
    radius = sigma * temporal_radius_scale(opacity_threshold) + float(extra_margin_s)
    selected = (times - radius <= float(t_end)) & (times + radius >= float(t_start))

    if include_static:
        is_static = np.asarray(
            flags.get("is_static", np.zeros((len(times),), dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if len(is_static) != len(times):
            raise ValueError(f"is_static mask length mismatch: {len(is_static)} != {len(times)}")
        selected |= is_static
    return np.ascontiguousarray(selected, dtype=bool)


def _regular_chunk_windows(
    *,
    t_max: float,
    chunk_duration_s: float,
    overlap_s: float,
) -> list[list[float]]:
    """Local copy of trainer chunk window generation for export-only use."""
    chunk_duration_s = float(chunk_duration_s)
    if chunk_duration_s <= 0.0:
        raise ValueError("--chunk-duration-seconds must be > 0")
    overlap_s = min(max(float(overlap_s), 0.0), 0.9 * chunk_duration_s)
    if t_max <= 0.0 or chunk_duration_s >= t_max:
        return [[0.0, float(t_max)]]

    stride = chunk_duration_s - overlap_s
    windows: list[list[float]] = []
    lo = 0.0
    while lo < t_max - 1e-9:
        hi = min(lo + chunk_duration_s, t_max)
        windows.append([float(lo), float(hi)])
        if hi >= t_max - 1e-9:
            break
        lo += stride
    return windows


def _ownership_bounds_for_windows(windows: list[list[float]]) -> list[float]:
    bounds: list[float] = []
    for previous, current in pairwise(windows):
        overlap_lo = max(float(previous[0]), float(current[0]))
        overlap_hi = min(float(previous[1]), float(current[1]))
        if overlap_hi < overlap_lo:
            bounds.append(0.5 * (float(previous[1]) + float(current[0])))
        else:
            bounds.append(0.5 * (overlap_lo + overlap_hi))
    return bounds


def inferred_checkpoint_frame_count(meta: dict[str, Any], frame_rate: float) -> int | None:
    """Infer the complete sequence length saved by the trainer."""
    candidates: list[int] = []
    if isinstance(meta.get("n_frames"), (int, np.integer)):
        candidates.append(int(meta["n_frames"]))

    trainer_windows = meta.get("trainerWindows")
    if isinstance(trainer_windows, list) and trainer_windows:
        try:
            last_time = max(float(window[1]) for window in trainer_windows)
            candidates.append(round(last_time * float(frame_rate)) + 1)
        except (TypeError, ValueError, IndexError):
            pass

    positive = [value for value in candidates if value > 0]
    return max(positive) if positive else None


def build_single_checkpoint_split_plan(
    *,
    meta: dict[str, Any],
    frame_rate: float,
    total_frames: int | None,
    chunk_duration_s: float | None,
    chunk_overlap_s: float,
    max_chunk_duration_s: float | None = None,
) -> tuple[list[list[float]], list[float], list[ClipSchedule], int, str]:
    """Build overlapping training ranges and non-overlapping playback cores."""
    fps = float(frame_rate)
    if fps <= 0.0:
        raise ValueError("frame rate must be positive")
    if chunk_duration_s is not None and max_chunk_duration_s is not None:
        raise ValueError(
            "pass --chunk-duration-seconds or --max-chunk-duration-seconds, not both"
        )

    trainer_windows = meta.get("trainerWindows")
    balanced = max_chunk_duration_s is not None
    if balanced:
        if total_frames is not None:
            frames_for_plan = int(total_frames)
        else:
            frames_for_plan = inferred_checkpoint_frame_count(meta, fps)
            if frames_for_plan is None:
                raise ValueError(
                    "cannot infer sequence length; pass --total-frames with "
                    "--max-chunk-duration-seconds"
                )
        chunk_duration_s = balanced_chunk_duration(
            float(max(frames_for_plan - 1, 0)) / fps, float(max_chunk_duration_s)
        )
    if chunk_duration_s is None:
        if not isinstance(trainer_windows, list) or not trainer_windows:
            raise ValueError(
                "checkpoint has no trainer chunk windows; pass --chunk-duration-seconds"
            )
        windows = [[float(window[0]), float(window[1])] for window in trainer_windows]
        source = "trainer_metrics"
    else:
        if total_frames is not None:
            inferred_frames = int(total_frames)
        else:
            inferred_frames = inferred_checkpoint_frame_count(meta, fps)
            if inferred_frames is None:
                raise ValueError(
                    "cannot infer sequence length; pass --total-frames with --chunk-duration-seconds"
                )
        t_max = float(max(inferred_frames - 1, 0)) / fps
        windows = _regular_chunk_windows(
            t_max=t_max,
            chunk_duration_s=float(chunk_duration_s),
            overlap_s=float(chunk_overlap_s),
        )
        source = "balanced_max_duration" if balanced else "fixed_duration"

    if not windows:
        raise ValueError("split plan contains no windows")
    for index, (lo, hi) in enumerate(windows):
        if lo < 0.0 or hi < lo:
            raise ValueError(f"invalid split window {index}: [{lo}, {hi}]")
        if index and lo > windows[index - 1][1] + 1e-9:
            raise ValueError(f"split windows contain a gap before window {index}")

    if total_frames is not None:
        n_frames = int(total_frames)
    else:
        inferred_frames = inferred_checkpoint_frame_count(meta, fps)
        n_frames = (
            inferred_frames
            if inferred_frames is not None
            else round(float(windows[-1][1]) * fps) + 1
        )
    if n_frames <= 0:
        raise ValueError("total frames must be positive")

    metric_bounds = meta.get("trainerOwnershipBounds")
    if (
        source == "trainer_metrics"
        and isinstance(metric_bounds, list)
        and len(metric_bounds) == len(windows) - 1
    ):
        bounds = [float(value) for value in metric_bounds]
    else:
        bounds = _ownership_bounds_for_windows(windows)

    core_bound_frames = [round(value * fps) for value in bounds]
    schedules: list[ClipSchedule] = []
    for index, (lo, hi) in enumerate(windows):
        train_start = max(0, round(lo * fps))
        # Trainer windows include the sample at t_hi. Convert that inclusive
        # endpoint to the packager's end-exclusive frame convention.
        train_end = min(n_frames, round(hi * fps) + 1)
        core_start = 0 if index == 0 else core_bound_frames[index - 1]
        core_end = n_frames if index == len(windows) - 1 else core_bound_frames[index]
        schedule = ClipSchedule(index, train_start, train_end, core_start, core_end)
        validate_schedule_item(schedule)
        schedules.append(schedule)

    return windows, bounds, schedules, n_frames, source


def playback_clock_frame_count(
    times: np.ndarray,
    *,
    frame_rate: float,
    schedule_frame_count: int,
) -> tuple[int, float | None, float | None, float | None]:
    """Keep legacy players from deriving playback speed from splat-center bounds.

    Some players compare the min/max of the selected ``times`` tensor with
    ``training.frameCount`` and rescale time when they differ. A temporal
    support slice legitimately contains centers outside its playback range,
    so that heuristic gives every slice a different clock. When necessary,
    expose a compatibility frame count matching the tensor span; the real
    scheduled frame count remains available as ``scheduleFrameCount``.
    """
    flat = np.asarray(times, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    fallback = max(1, int(schedule_frame_count))
    if finite.size == 0:
        return fallback, None, None, None

    time_min = float(finite.min())
    time_max = float(finite.max())
    span = max(0.0, time_max - time_min)
    if span <= 0.0:
        return fallback, time_min, time_max, 1.0

    expected_span = max(float(fallback - 1) / float(frame_rate), 1e-6)
    ratio = span / expected_span
    if 0.85 <= ratio <= 1.15:
        return fallback, time_min, time_max, ratio

    ideal = max(2, int(round(span * float(frame_rate))) + 1)
    guarded_expected_span = max(float(ideal - 1) / float(frame_rate), 1e-6)
    return ideal, time_min, time_max, span / guarded_expected_span


def build_runtime_clip_from_checkpoint(
    *,
    builder: RuntimeBinaryBuilder,
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
    flags: dict[str, np.ndarray],
    meta: dict[str, Any],
    schedule: ClipSchedule,
    frame_rate: float,
    dtype_policy: str,
    static_export_mode: str,
    fade_in: int,
    fade_out: int,
    fade_curve: str,
    transition_padding: int,
    preload_frames: int,
    keep_alive_frames: int,
    is_first: bool,
    is_last: bool,
    static_tensors_override: dict[str, dict[str, Any]] | None = None,
    static_reused_from_clip_id: str | None = None,
    static_storage_mode: str = "per-clip",
    global_static_id: str | None = None,
    global_static_count: int | None = None,
    global_static_sh_coeff_count: int | None = None,
) -> dict[str, Any]:
    clip_id = f"clip_{schedule.index:04d}"

    active_start = max(schedule.train_start, schedule.core_start - transition_padding)
    active_end = min(schedule.train_end, schedule.core_end + transition_padding)
    actual_fade_in = 0 if is_first else min(fade_in, max(0, schedule.core_start - active_start))
    actual_fade_out = 0 if is_last else min(fade_out, max(0, active_end - schedule.core_end))

    count = int(meta["count"])
    sh_coeff_count = int(meta["shCoeffCount"])
    sh_degree = int(meta["shDegree"])

    if static_export_mode == "split":
        static_mask = np.asarray(flags.get("is_static", np.zeros((count,), dtype=bool)), dtype=bool).reshape(-1)
        if len(static_mask) != count:
            raise ValueError(f"{checkpoint_path}: is_static mask length mismatch: {len(static_mask)} != {count}")
        dynamic_mask = ~static_mask
    elif static_export_mode == "all-dynamic":
        static_mask = np.zeros((count,), dtype=bool)
        dynamic_mask = np.ones((count,), dtype=bool)
    else:
        raise ValueError(f"Unsupported static export mode: {static_export_mode}")

    static_arrays = _slice_arrays(arrays, STATIC_KEYS, static_mask)
    dynamic_arrays = _slice_arrays(arrays, REQUIRED_KEYS, dynamic_mask)

    dynamic_tensors = add_array_group_to_builder(
        builder=builder,
        arrays=dynamic_arrays,
        clip_id=clip_id,
        group_name="dynamic",
        dtype_policy=dtype_policy,
    )

    if static_storage_mode not in ("per-clip", "global"):
        raise ValueError(f"Unsupported static_storage_mode: {static_storage_mode}")

    static_reused = static_tensors_override is not None
    static_is_global = static_storage_mode == "global"
    if static_is_global:
        static_tensors = {}
        static_reused = True
        if global_static_id is None:
            raise ValueError("global static mode requires global_static_id")
    elif static_tensors_override is not None:
        static_tensors = copy.deepcopy(static_tensors_override)
    else:
        static_tensors = add_array_group_to_builder(
            builder=builder,
            arrays=static_arrays,
            clip_id=clip_id,
            group_name="static",
            dtype_policy=dtype_policy,
        )

    flag_tensors: dict[str, dict[str, Any]] = {}
    for name in OPTIONAL_FLAG_KEYS:
        arr = np.asarray(flags.get(name, np.zeros((count,), dtype=bool)), dtype=np.uint8).reshape(-1)
        logical_path = f"clips/{clip_id}/flags/{name}.u8"
        flag_tensors[name] = builder.add_tensor(
            logical_path=logical_path,
            dtype="uint8",
            shape=[int(arr.shape[0])],
            data=encode_array(arr, "uint8"),
            clip_id=clip_id,
            group="flags",
            name=name,
        )

    local_static_count = int(static_mask.sum())
    static_count = int(global_static_count) if static_is_global and global_static_count is not None else local_static_count
    static_sh_coeff_count = (
        int(global_static_sh_coeff_count)
        if static_is_global and global_static_sh_coeff_count is not None
        else sh_coeff_count
    )
    dynamic_count = int(dynamic_mask.sum())
    model_count = int(static_count + dynamic_count) if static_is_global else count

    return {
        "id": clip_id,
        "index": int(schedule.index),
        "path": f"clips/{clip_id}",
        "source": {
            "kind": "checkpoint",
            "path": os.fspath(checkpoint_path),
            "checkpointStep": meta.get("step"),
        },
        "global": {
            "activeFrameStart": int(active_start),
            "activeFrameEndExclusive": int(active_end),
            "coreFrameStart": int(schedule.core_start),
            "coreFrameEndExclusive": int(schedule.core_end),
            "frameStart": int(schedule.core_start),
            "frameEndExclusive": int(schedule.core_end),
            "activeTimeStart": float(active_start) / frame_rate,
            "activeTimeEnd": float(active_end) / frame_rate,
            "coreTimeStart": float(schedule.core_start) / frame_rate,
            "coreTimeEnd": float(schedule.core_end) / frame_rate,
        },
        "local": {
            "activeFrameStart": int(active_start - schedule.train_start),
            "activeFrameEndExclusive": int(active_end - schedule.train_start),
            "coreFrameStart": int(schedule.local_core_start),
            "coreFrameEndExclusive": int(schedule.local_core_end),
            "frameStart": int(schedule.local_core_start),
            "frameEndExclusive": int(schedule.local_core_end),
            "timeStart": float(schedule.local_core_start) / frame_rate,
            "timeEnd": float(schedule.local_core_end) / frame_rate,
        },
        "training": {
            "globalFrameStart": int(schedule.train_start),
            "globalFrameEndExclusive": int(schedule.train_end),
            "localFrameStart": 0,
            "localFrameEndExclusive": int(schedule.train_frames),
            "frameCount": int(schedule.train_frames),
            "hasPaddingFrames": bool(schedule.local_core_start > 0 or schedule.train_end > schedule.core_end),
            "paddingBeforeFrames": int(schedule.local_core_start),
            "paddingAfterFrames": int(schedule.train_end - schedule.core_end),
        },
        "render": {
            "frameCount": int(schedule.core_frames),
            "activeFrameCount": int(active_end - active_start),
            "transitionMode": "opacity_fade" if (actual_fade_in or actual_fade_out) else "hard_cut",
            "blendMode": "weighted_opacity",
            "fadeInFrames": int(actual_fade_in),
            "fadeOutFrames": int(actual_fade_out),
            "opacityFadeInFrames": int(actual_fade_in),
            "opacityFadeOutFrames": int(actual_fade_out),
            "fadeCurve": fade_curve,
            "transitionPaddingFrames": int(transition_padding),
            "preloadFrames": int(preload_frames),
            "keepAliveFrames": int(keep_alive_frames),
        },
        "model": {
            "count": model_count,
            "sourceCheckpointCount": count,
            "staticCount": static_count,
            "localStaticCount": local_static_count,
            "dynamicCount": dynamic_count,
            "dynamicProtectCount": int(meta.get("dynamicProtectCount", 0)),
            "shCoeffCount": sh_coeff_count,
            "shDegree": sh_degree,
            "motionModel": "trigs.se3_bezier_v1",
            "tensorLayout": "split_static_dynamic" if static_export_mode == "split" else "split_dynamic",
            "staticExportMode": static_export_mode,
            "staticStorage": "global" if static_is_global else ("reused" if static_reused else "owned"),
        },
        "dynamic": {
            "count": dynamic_count,
            "shCoeffCount": sh_coeff_count,
            "tensors": dynamic_tensors,
        },
        "static": {
            "count": static_count,
            "shCoeffCount": static_sh_coeff_count,
            "storage": "global" if static_is_global else ("reused" if static_reused else "owned"),
            "sourceClipId": static_reused_from_clip_id if static_reused else clip_id,
            "sourceStaticId": global_static_id if static_is_global else None,
            "tensors": static_tensors,
        },
        "flags": {
            "count": count,
            "tensors": flag_tensors,
        },
    }


def convert_from_checkpoints(args: argparse.Namespace) -> int:
    if args.frame_rate is None or args.frame_rate <= 0:
        raise SystemExit("--frame-rate must be positive")
    checkpoints = read_checkpoint_paths(args)
    if not checkpoints:
        raise SystemExit("Pass at least one --checkpoint or --checkpoint-list")
    if len(args.ranges) != len(checkpoints):
        raise SystemExit(f"Expected exactly {len(checkpoints)} --range values, got {len(args.ranges)}")

    schedules = [parse_range(raw, i) for i, raw in enumerate(args.ranges)]
    static_source_clip = int(args.static_source_clip)
    if static_source_clip < 0 or static_source_clip >= len(checkpoints):
        raise SystemExit(
            f"--static-source-clip must be in [0,{len(checkpoints) - 1}], got {static_source_clip}"
        )
    total_frames = int(args.total_frames) if args.total_frames is not None else max(s.core_end for s in schedules)
    if total_frames <= 0:
        raise SystemExit("total frames must be positive")

    fade_in = int(args.fade_in_frames if args.fade_in_frames is not None else args.fade_frames)
    fade_out = int(args.fade_out_frames if args.fade_out_frames is not None else args.fade_frames)
    if fade_in < 0 or fade_out < 0:
        raise SystemExit("fade frames must be >= 0")
    transition_padding = int(args.transition_padding_frames if args.transition_padding_frames is not None else max(fade_in, fade_out))

    builder = RuntimeBinaryBuilder(alignment=args.alignment, include_hashes=args.tensor_hashes)
    runtime_clips: list[dict[str, Any]] = []

    global_static_group: dict[str, Any] | None = None
    global_static_id = "static_0000"
    global_static_count: int | None = None
    global_static_sh_coeff_count: int | None = None

    if str(args.static_storage) == "global" and str(args.static_export_mode) == "split":
        static_ckpt_path = checkpoints[static_source_clip]
        if not static_ckpt_path.is_file():
            raise SystemExit(f"Static source checkpoint not found: {static_ckpt_path}")
        static_arrays_all, static_flags_all, static_meta = load_checkpoint_arrays(static_ckpt_path)
        static_arrays, static_count, static_sh_coeff_count, static_sh_degree = extract_static_arrays_from_checkpoint_payload(
            checkpoint_path=static_ckpt_path,
            arrays=static_arrays_all,
            flags=static_flags_all,
            meta=static_meta,
            static_export_mode=str(args.static_export_mode),
        )
        static_tensors = add_global_static_group_to_builder(
            builder=builder,
            arrays=static_arrays,
            dtype_policy=str(args.tensor_dtype),
            static_id=global_static_id,
        )
        global_static_count = int(static_count)
        global_static_sh_coeff_count = int(static_sh_coeff_count)
        global_static_group = {
            "id": global_static_id,
            "storage": "global",
            "source": {
                "kind": "checkpoint",
                "clipIndex": int(static_source_clip),
                "clipId": f"clip_{static_source_clip:04d}",
                "path": os.fspath(static_ckpt_path),
                "checkpointStep": static_meta.get("step"),
            },
            "global": {
                "frameStart": 0,
                "frameEndExclusive": int(total_frames),
                "timeStart": 0.0,
                "timeEnd": float(total_frames) / float(args.frame_rate),
            },
            "count": int(static_count),
            "shCoeffCount": int(static_sh_coeff_count),
            "shDegree": int(static_sh_degree),
            "tensors": static_tensors,
        }
        print(
            f"  global static: source clip {static_source_clip} "
            f"({static_ckpt_path}), rows={static_count:,}"
        )

    shared_static_tensors: dict[str, dict[str, Any]] | None = None
    shared_static_signature: str | None = None
    shared_static_clip_id: str | None = None

    color_reference_anchor: np.ndarray | None = None
    color_previous_anchor: np.ndarray | None = None
    color_normalization_records: list[dict[str, Any]] = []
    overlap_color_calibration_records: list[dict[str, Any]] = []
    previous_arrays_for_overlap: dict[str, np.ndarray] | None = None
    previous_flags_for_overlap: dict[str, np.ndarray] | None = None
    previous_schedule_for_overlap: ClipSchedule | None = None

    print(f"TRiGS Runtime Binary Packager: {VERSION}")
    print(f"  output: {args.output}")
    print(f"  checkpoints: {len(checkpoints)}")
    print(f"  fps: {args.frame_rate}")
    print(f"  dtype policy: {args.tensor_dtype}")
    print(f"  static export mode: {args.static_export_mode}")
    print(f"  static storage: {args.static_storage}")
    print(f"  static source clip: {static_source_clip}")
    print(f"  shared static: {args.shared_static}")
    print(f"  color normalize: {args.color_normalize}")
    print(f"  overlap color calibrate: {args.overlap_color_calibrate}")

    for i, (ckpt_path, sched) in enumerate(zip(checkpoints, schedules)):
        if not ckpt_path.is_file():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")
        arrays, flags, meta = load_checkpoint_arrays(ckpt_path)

        if str(args.overlap_color_calibrate) != "none":
            if (
                i > 0
                and previous_arrays_for_overlap is not None
                and previous_flags_for_overlap is not None
                and previous_schedule_for_overlap is not None
            ):
                overlap_record = apply_overlap_color_calibration(
                    previous_arrays_for_overlap,
                    previous_flags_for_overlap,
                    previous_schedule_for_overlap,
                    arrays,
                    flags,
                    sched,
                    frame_rate=float(args.frame_rate),
                    mode=str(args.overlap_color_calibrate),
                    apply_to=str(args.overlap_color_calibrate_apply),
                    max_gain=float(args.overlap_color_calibrate_max_gain),
                    strength=float(args.overlap_color_calibrate_strength),
                    min_samples=int(args.overlap_color_calibrate_min_samples),
                    min_alpha_percentile=float(args.overlap_color_calibrate_min_alpha_percentile),
                )
                overlap_record.update({"clipIndex": int(i), "clipId": f"clip_{i:04d}"})
                overlap_color_calibration_records.append(overlap_record)
                if overlap_record.get("skipped"):
                    print(
                        f"  clip_{i:04d} overlap color calib: skipped "
                        f"reason={overlap_record.get('reason')} "
                        f"overlap=[{overlap_record.get('overlapStart')},{overlap_record.get('overlapEnd')})"
                    )
                else:
                    g = overlap_record["gain"]
                    print(
                        f"  clip_{i:04d} overlap color calib: "
                        f"overlap=[{overlap_record['overlapStart']},{overlap_record['overlapEnd']}) "
                        f"samples ref/current={overlap_record['referenceSamples']:,}/{overlap_record['currentSamples']:,} "
                        f"gain=[{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}] "
                        f"mode={args.overlap_color_calibrate} apply={args.overlap_color_calibrate_apply}"
                    )
            else:
                overlap_color_calibration_records.append(
                    {
                        "clipIndex": int(i),
                        "clipId": f"clip_{i:04d}",
                        "enabled": True,
                        "skipped": True,
                        "reason": "first_clip_reference",
                        "gain": [1.0, 1.0, 1.0],
                    }
                )
                print(f"  clip_{i:04d} overlap color calib: reference clip, gain=[1.0000, 1.0000, 1.0000]")

        if str(args.color_normalize) != "none":
            if str(args.color_normalize_reference) == "first":
                reference_anchor = color_reference_anchor
            else:
                reference_anchor = color_previous_anchor

            before_anchor, ref_anchor, gain, sample_count, used_mode = apply_clip_color_normalization(
                arrays,
                flags,
                mode=str(args.color_normalize),
                reference_anchor=reference_anchor,
                max_gain=float(args.color_normalize_max_gain),
                strength=float(args.color_normalize_strength),
                apply_to=str(args.color_normalize_apply),
                min_samples=int(args.color_normalize_min_samples),
            )

            after_anchor, _, _ = _robust_sh0_anchor(
                arrays,
                flags,
                mode=str(args.color_normalize),
                min_samples=int(args.color_normalize_min_samples),
            )

            if color_reference_anchor is None:
                color_reference_anchor = before_anchor.copy()
            color_previous_anchor = after_anchor.copy()

            color_normalization_records.append(
                {
                    "clipIndex": int(i),
                    "clipId": f"clip_{i:04d}",
                    "mode": str(args.color_normalize),
                    "usedMode": used_mode,
                    "reference": str(args.color_normalize_reference),
                    "applyTo": str(args.color_normalize_apply),
                    "sampleCount": int(sample_count),
                    "anchorBefore": [float(v) for v in before_anchor],
                    "referenceAnchor": [float(v) for v in ref_anchor],
                    "gain": [float(v) for v in gain],
                }
            )
            print(
                f"  clip_{i:04d} color norm: samples={sample_count:,} "
                f"gain=[{gain[0]:.4f}, {gain[1]:.4f}, {gain[2]:.4f}] "
                f"mode={used_mode} apply={args.color_normalize_apply}"
            )

        static_tensors_override = None
        static_reused_from_clip_id = None
        current_static_signature = None

        if str(args.static_storage) != "global" and str(args.shared_static) != "off" and str(args.static_export_mode) == "split":
            count = int(meta["count"])
            static_mask_for_reuse = np.asarray(flags.get("is_static", np.zeros((count,), dtype=bool)), dtype=bool).reshape(-1)
            if len(static_mask_for_reuse) != count:
                raise ValueError(f"{ckpt_path}: is_static mask length mismatch: {len(static_mask_for_reuse)} != {count}")
            static_arrays_for_reuse = _slice_arrays(arrays, STATIC_KEYS, static_mask_for_reuse)
            current_static_signature = static_group_signature(static_arrays_for_reuse, str(args.tensor_dtype))

            if shared_static_tensors is not None:
                if str(args.shared_static) == "exact":
                    if current_static_signature == shared_static_signature:
                        static_tensors_override = shared_static_tensors
                        static_reused_from_clip_id = shared_static_clip_id
                elif str(args.shared_static) == "first":
                    if not static_tensor_specs_compatible(shared_static_tensors, static_arrays_for_reuse, str(args.tensor_dtype)):
                        raise ValueError(
                            f"{ckpt_path}: --shared-static first requested, but static tensor shapes/dtypes "
                            f"do not match {shared_static_clip_id}. Use --shared-static off or exact."
                        )
                    static_tensors_override = shared_static_tensors
                    static_reused_from_clip_id = shared_static_clip_id

        clip = build_runtime_clip_from_checkpoint(
            builder=builder,
            checkpoint_path=ckpt_path,
            arrays=arrays,
            flags=flags,
            meta=meta,
            schedule=sched,
            frame_rate=float(args.frame_rate),
            dtype_policy=str(args.tensor_dtype),
            static_export_mode=str(args.static_export_mode),
            fade_in=fade_in,
            fade_out=fade_out,
            fade_curve=str(args.fade_curve),
            transition_padding=transition_padding,
            preload_frames=int(args.preload_frames),
            keep_alive_frames=int(args.keep_alive_frames),
            is_first=(i == 0),
            is_last=(i == len(schedules) - 1),
            static_tensors_override=static_tensors_override,
            static_reused_from_clip_id=(
                f"clip_{static_source_clip:04d}"
                if str(args.static_storage) == "global"
                else static_reused_from_clip_id
            ),
            static_storage_mode=str(args.static_storage),
            global_static_id=global_static_id if str(args.static_storage) == "global" else None,
            global_static_count=global_static_count,
            global_static_sh_coeff_count=global_static_sh_coeff_count,
        )

        if (
            str(args.static_storage) != "global"
            and str(args.shared_static) != "off"
            and str(args.static_export_mode) == "split"
            and shared_static_tensors is None
        ):
            shared_static_tensors = copy.deepcopy(clip["static"]["tensors"])
            shared_static_signature = current_static_signature
            shared_static_clip_id = str(clip["id"])
            print(f"  {clip['id']}: registered shared static reference")
        elif clip.get("static", {}).get("storage") == "reused":
            print(f"  {clip['id']}: reusing static tensors from {clip['static'].get('sourceClipId')}")
        runtime_clips.append(clip)
        print(
            f"  {clip['id']}: total={clip['model']['count']:,} "
            f"static={clip['model']['staticCount']:,} dynamic={clip['model']['dynamicCount']:,} "
            f"active=[{clip['global']['activeFrameStart']},{clip['global']['activeFrameEndExclusive']}) "
            f"core=[{clip['global']['coreFrameStart']},{clip['global']['coreFrameEndExclusive']})"
        )

        # Keep the corrected current clip as the reference for the next boundary.
        previous_arrays_for_overlap = arrays
        previous_flags_for_overlap = flags
        previous_schedule_for_overlap = sched

    timeline = {
        "frameRate": float(args.frame_rate),
        "totalFrames": int(total_frames),
        "durationSeconds": float(total_frames) / float(args.frame_rate),
        "timeUnit": "seconds",
        "frameIndexing": "zero_based_end_exclusive",
    }
    packaging = {
        "runtimePackagerVersion": VERSION,
        "sourceAuthoringFormat": "checkpoint-native",
        "staticExportMode": str(args.static_export_mode),
        "staticStorage": str(args.static_storage),
        "staticSourceClip": int(static_source_clip),
        "globalStaticId": global_static_id if global_static_group is not None else None,
        "tensorDtypePolicy": str(args.tensor_dtype),
        "sharedStatic": str(args.shared_static),
        "colorNormalize": str(args.color_normalize),
        "colorNormalizeReference": str(args.color_normalize_reference),
        "colorNormalizeApply": str(args.color_normalize_apply),
        "colorNormalizeMaxGain": float(args.color_normalize_max_gain),
        "colorNormalizeStrength": float(args.color_normalize_strength),
        "colorNormalizationRecords": color_normalization_records,
        "overlapColorCalibrate": str(args.overlap_color_calibrate),
        "overlapColorCalibrateApply": str(args.overlap_color_calibrate_apply),
        "overlapColorCalibrateMaxGain": float(args.overlap_color_calibrate_max_gain),
        "overlapColorCalibrateStrength": float(args.overlap_color_calibrate_strength),
        "overlapColorCalibrationRecords": overlap_color_calibration_records,
        "payloadEncoding": "single-file-raw-aligned",
        "alignment": int(args.alignment),
    }
    metadata: dict[str, Any] = {
        "createdBy": Path(__file__).name,
        "packagerVersion": VERSION,
    }
    if args.name:
        metadata["name"] = args.name
    if args.description:
        metadata["description"] = args.description
    playback = {
        "loop": bool(args.loop),
        "chunkSelection": "activeFrameRange",
        "transitionMode": "opacity_fade" if any(c["render"]["transitionMode"] == "opacity_fade" for c in runtime_clips) else "hard_cut",
        "blendMode": "weighted_opacity",
    }

    runtime_manifest = default_runtime_manifest_base(
        source_kind="trainer-checkpoints",
        source_path=None,
        timeline=timeline,
        clips=runtime_clips,
        packaging=packaging,
        metadata=metadata,
        playback=playback,
    )
    if global_static_group is not None:
        runtime_manifest["static"] = global_static_group
        runtime_manifest.setdefault("semantics", {})["staticStorage"] = "global"
        runtime_manifest["semantics"]["globalStaticGroup"] = True
        runtime_manifest["semantics"]["clipsReferenceGlobalStatic"] = True
    written = builder.write(
        output=Path(args.output),
        manifest=runtime_manifest,
        overwrite=args.overwrite,
        manifest_output=args.manifest_output,
    )
    print_summary(Path(args.output), written)
    return 0


def convert_from_single_checkpoint(args: argparse.Namespace) -> int:
    """Partition one assembled long-sequence checkpoint into runtime clips."""
    if args.frame_rate is None or args.frame_rate <= 0:
        raise SystemExit("--frame-rate must be positive")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    arrays, flags, meta = load_checkpoint_arrays(checkpoint_path)
    checkpoint_fps = meta.get("fps")
    if checkpoint_fps is not None and not math.isclose(
        float(checkpoint_fps), float(args.frame_rate), rel_tol=0.0, abs_tol=1e-6
    ):
        raise SystemExit(
            f"--frame-rate {args.frame_rate:g} does not match checkpoint fps "
            f"{float(checkpoint_fps):g}"
        )
    inferred_total_frames = inferred_checkpoint_frame_count(
        meta, float(args.frame_rate)
    )
    if (
        args.total_frames is not None
        and inferred_total_frames is not None
        and int(args.total_frames) < inferred_total_frames
        and not args.allow_truncate
    ):
        missing_frames = inferred_total_frames - int(args.total_frames)
        raise SystemExit(
            f"--total-frames {int(args.total_frames)} would truncate the trainer timeline "
            f"({inferred_total_frames} frames inferred from the checkpoint; "
            f"{missing_frames} frames would be dropped). Omit --total-frames to export the "
            "complete sequence, or pass --allow-truncate if this is intentional."
        )
    try:
        windows, ownership_bounds, schedules, total_frames, plan_source = (
            build_single_checkpoint_split_plan(
                meta=meta,
                frame_rate=float(args.frame_rate),
                total_frames=args.total_frames,
                chunk_duration_s=args.chunk_duration_seconds,
                chunk_overlap_s=float(args.chunk_overlap_seconds),
                max_chunk_duration_s=args.max_chunk_duration_seconds,
            )
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    threshold, threshold_source = resolve_temporal_threshold(
        args.temporal_threshold, meta
    )
    if threshold_source == "trainer_default":
        print(
            "WARNING: checkpoint records no temporal threshold (no trainer metric, no "
            f"config); using the trainer default {threshold:g}. Pass --temporal-threshold "
            "if the run changed --config.temporal-active-threshold, or package the copy "
            "written by prepare_export_checkpoint.py.",
            file=sys.stderr,
        )
    if not 0.0 < threshold < 1.0:
        raise SystemExit("--temporal-threshold must be between 0 and 1")
    extra_margin_s = float(args.temporal_extra_margin_seconds)
    if extra_margin_s < 0.0:
        raise SystemExit("--temporal-extra-margin-seconds must be >= 0")

    fade_in = int(
        args.fade_in_frames if args.fade_in_frames is not None else args.fade_frames
    )
    fade_out = int(
        args.fade_out_frames if args.fade_out_frames is not None else args.fade_frames
    )
    if fade_in < 0 or fade_out < 0:
        raise SystemExit("fade frames must be >= 0")
    transition_padding = int(
        args.transition_padding_frames
        if args.transition_padding_frames is not None
        else max(fade_in, fade_out)
    )

    count = int(meta["count"])
    is_static = np.asarray(
        flags.get("is_static", np.zeros((count,), dtype=bool)), dtype=bool
    ).reshape(-1)
    if len(is_static) != count:
        raise SystemExit(f"is_static mask length mismatch: {len(is_static)} != {count}")
    static_count = int(is_static.sum())
    dynamic_count = int(count - static_count)

    builder = RuntimeBinaryBuilder(
        alignment=args.alignment, include_hashes=args.tensor_hashes
    )
    global_static_id = "static_0000"
    global_static_arrays = _slice_arrays(arrays, STATIC_KEYS, is_static)
    global_static_tensors = add_global_static_group_to_builder(
        builder=builder,
        arrays=global_static_arrays,
        dtype_policy=str(args.tensor_dtype),
        static_id=global_static_id,
    )
    global_static_group = {
        "id": global_static_id,
        "storage": "global",
        "source": {
            "kind": "checkpoint",
            "path": os.fspath(checkpoint_path),
            "checkpointStep": meta.get("step"),
        },
        "global": {
            "frameStart": 0,
            "frameEndExclusive": int(total_frames),
            "timeStart": 0.0,
            "timeEnd": float(total_frames) / float(args.frame_rate),
        },
        "count": static_count,
        "shCoeffCount": int(meta["shCoeffCount"]),
        "shDegree": int(meta["shDegree"]),
        "tensors": global_static_tensors,
    }

    runtime_clips: list[dict[str, Any]] = []
    stored_dynamic_rows = 0
    radius_scale = temporal_radius_scale(threshold)

    print(f"TRiGS Runtime Binary Packager: {VERSION}")
    print(f"  output: {args.output}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  split plan: {plan_source} ({len(windows)} clips)")
    print(
        f"  source rows: {count:,} "
        f"(static={static_count:,}, dynamic={dynamic_count:,})"
    )
    print(
        f"  temporal support: threshold={threshold:g} "
        f"radius={radius_scale:.6f}*sigma + {extra_margin_s:g}s"
    )

    time_domain = str(args.time_domain)
    for window, selection_schedule in zip(windows, schedules):
        t_lo, t_hi = float(window[0]), float(window[1])
        support_lo, support_hi = t_lo, t_hi
        if time_domain == "global" and transition_padding > 0:
            active_start_frame = max(
                0, selection_schedule.core_start - transition_padding
            )
            active_end_frame = min(
                total_frames, selection_schedule.core_end + transition_padding
            )
            support_lo = min(
                support_lo, float(active_start_frame) / float(args.frame_rate)
            )
            support_hi = max(
                support_hi,
                float(max(active_end_frame - 1, 0)) / float(args.frame_rate),
            )
        selected = temporal_support_mask(
            arrays,
            flags,
            t_start=support_lo,
            t_end=support_hi,
            opacity_threshold=threshold,
            extra_margin_s=extra_margin_s,
            include_static=False,
        )
        selected &= ~is_static
        selected_count = int(selected.sum())
        stored_dynamic_rows += selected_count

        clip_arrays = _slice_arrays(arrays, REQUIRED_KEYS, selected)
        if time_domain == "clip-local":
            # Match the established multi-checkpoint format: tensor time is
            # relative to the clip's training anchor, while the manifest maps
            # the global timeline onto that local clock exactly once.
            time_origin_frame = int(selection_schedule.train_start)
            clip_arrays["times"] = np.ascontiguousarray(
                clip_arrays["times"]
                - np.float32(float(time_origin_frame) / float(args.frame_rate)),
                dtype=np.float32,
            )
            runtime_schedule = selection_schedule
        else:
            time_origin_frame = 0
            clip_arrays["times"] = np.ascontiguousarray(
                clip_arrays["times"], dtype=np.float32
            )
            runtime_schedule = ClipSchedule(
                selection_schedule.index,
                0,
                total_frames,
                selection_schedule.core_start,
                selection_schedule.core_end,
            )
        clip_flags = {
            name: np.ascontiguousarray(
                np.asarray(flags.get(name, np.zeros((count,), dtype=bool)))[selected],
                dtype=bool,
            )
            for name in OPTIONAL_FLAG_KEYS
        }
        clip_meta = dict(meta)
        clip_meta.update(
            {
                "count": selected_count,
                "staticCount": 0,
                "dynamicCount": selected_count,
                "dynamicProtectCount": int(
                    np.asarray(
                        flags.get("dynamic_protect", np.zeros((count,), dtype=bool)),
                        dtype=bool,
                    ).reshape(-1)[selected].sum()
                ),
            }
        )

        clip = build_runtime_clip_from_checkpoint(
            builder=builder,
            checkpoint_path=checkpoint_path,
            arrays=clip_arrays,
            flags=clip_flags,
            meta=clip_meta,
            schedule=runtime_schedule,
            frame_rate=float(args.frame_rate),
            dtype_policy=str(args.tensor_dtype),
            static_export_mode="split",
            fade_in=fade_in,
            fade_out=fade_out,
            fade_curve=str(args.fade_curve),
            transition_padding=transition_padding,
            preload_frames=int(args.preload_frames),
            keep_alive_frames=int(args.keep_alive_frames),
            is_first=(selection_schedule.index == 0),
            is_last=(selection_schedule.index == len(schedules) - 1),
            static_tensors_override=None,
            static_reused_from_clip_id=None,
            static_storage_mode="global",
            global_static_id=global_static_id,
            global_static_count=static_count,
            global_static_sh_coeff_count=int(meta["shCoeffCount"]),
        )
        scheduled_frame_count = int(clip["training"]["frameCount"])
        (
            clock_frame_count,
            tensor_time_min,
            tensor_time_max,
            clock_span_ratio,
        ) = playback_clock_frame_count(
            clip_arrays["times"],
            frame_rate=float(args.frame_rate),
            schedule_frame_count=scheduled_frame_count,
        )
        clip["training"]["scheduleFrameCount"] = scheduled_frame_count
        clip["training"]["frameCount"] = clock_frame_count
        clip["training"]["frameCountPurpose"] = "playback_clock_compatibility"
        clip["source"]["sourceCheckpointCount"] = count
        clip["source"]["temporalSelection"] = {
            "windowTimeStart": t_lo,
            "windowTimeEnd": t_hi,
            "supportTimeStart": support_lo,
            "supportTimeEnd": support_hi,
            "opacityThreshold": threshold,
            "radiusScale": radius_scale,
            "extraMarginSeconds": extra_margin_s,
            "selectedDynamicRows": selected_count,
            "selectionFrameStart": int(selection_schedule.train_start),
            "selectionFrameEndExclusive": int(selection_schedule.train_end),
            "timeDomain": time_domain,
            "timesRebasedBySeconds": float(time_origin_frame)
            / float(args.frame_rate),
            "tensorTimeMin": tensor_time_min,
            "tensorTimeMax": tensor_time_max,
            "playbackClockSpanRatio": clock_span_ratio,
            "scheduledTrainingFrameCount": scheduled_frame_count,
            "playbackClockFrameCount": clock_frame_count,
        }
        runtime_clips.append(clip)
        print(
            f"  {clip['id']}: dynamic={selected_count:,} "
            f"selection=[{selection_schedule.train_start},"
            f"{selection_schedule.train_end}) "
            f"core=[{selection_schedule.core_start},"
            f"{selection_schedule.core_end}) time={time_domain} "
            f"clockFrames={clock_frame_count}"
        )

    timeline = {
        "frameRate": float(args.frame_rate),
        "totalFrames": int(total_frames),
        "durationSeconds": float(total_frames) / float(args.frame_rate),
        "timeUnit": "seconds",
        "frameIndexing": "zero_based_end_exclusive",
    }
    packaging = {
        "runtimePackagerVersion": VERSION,
        "sourceAuthoringFormat": "single-checkpoint-auto-split",
        "tensorDtypePolicy": str(args.tensor_dtype),
        "staticExportMode": "split",
        "staticStorage": "global",
        "globalStaticId": global_static_id,
        "payloadEncoding": "single-file-raw-aligned",
        "alignment": int(args.alignment),
        "autoSplit": {
            "planSource": plan_source,
            "trainerWindows": windows,
            "ownershipBounds": ownership_bounds,
            "opacityThreshold": threshold,
            "opacityThresholdSource": threshold_source,
            "radiusScale": radius_scale,
            "extraMarginSeconds": extra_margin_s,
            "timeDomain": time_domain,
            "timesRebasedToClipTrainingOrigin": time_domain == "clip-local",
            "playbackClockCompatibility": "training_frame_count_matches_tensor_center_span",
            "sourceRows": count,
            "sourceStaticRows": static_count,
            "sourceDynamicRows": dynamic_count,
            "storedDynamicRows": stored_dynamic_rows,
            "dynamicDuplicationFactor": (
                float(stored_dynamic_rows) / float(dynamic_count)
                if dynamic_count > 0
                else 0.0
            ),
        },
    }
    metadata: dict[str, Any] = {
        "createdBy": Path(__file__).name,
        "packagerVersion": VERSION,
    }
    if args.name:
        metadata["name"] = args.name
    if args.description:
        metadata["description"] = args.description
    playback = {
        "loop": bool(args.loop),
        "chunkSelection": "activeFrameRange",
        "transitionMode": (
            "opacity_fade" if fade_in > 0 or fade_out > 0 else "hard_cut"
        ),
        "blendMode": "weighted_opacity",
    }

    runtime_manifest = default_runtime_manifest_base(
        source_kind="single-trainer-checkpoint-auto-split",
        source_path=os.fspath(checkpoint_path),
        timeline=timeline,
        clips=runtime_clips,
        packaging=packaging,
        metadata=metadata,
        playback=playback,
    )
    runtime_manifest["static"] = global_static_group
    runtime_manifest["semantics"].update(
        {
            "staticStorage": "global",
            "globalStaticGroup": True,
            "clipsReferenceGlobalStatic": True,
            "temporalCullThreshold": threshold,
            "temporalCullRadiusScale": radius_scale,
            "temporalCullHasFiniteSupport": True,
            "temporalOpacityMode": "smooth_cull_renormalized",
            "dynamicTimesDomain": time_domain,
        }
    )
    written = builder.write(
        output=Path(args.output),
        manifest=runtime_manifest,
        overwrite=args.overwrite,
        manifest_output=args.manifest_output,
    )
    print(
        f"  stored dynamic rows: {stored_dynamic_rows:,} "
        f"({packaging['autoSplit']['dynamicDuplicationFactor']:.3f}x source dynamic)"
    )
    print_summary(Path(args.output), written)
    return 0


# -----------------------------------------------------------------------------
# inspect path
# -----------------------------------------------------------------------------

def read_trigsbin_header(f: BinaryIO) -> dict[str, Any]:
    raw = f.read(HEADER_BYTES)
    if len(raw) != HEADER_BYTES:
        raise ValueError("file too small for TRIGSBIN header")
    magic, version, header_bytes, manifest_offset, manifest_length, payload_offset, payload_length, alignment, reserved = HEADER_STRUCT.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"not a TRIGSBIN file: magic={magic!r}")
    if header_bytes != HEADER_BYTES:
        raise ValueError(f"unsupported header size: {header_bytes}")
    return {
        "magic": magic.decode("ascii"),
        "version": int(version),
        "headerBytes": int(header_bytes),
        "manifestOffset": int(manifest_offset),
        "manifestLength": int(manifest_length),
        "payloadOffset": int(payload_offset),
        "payloadLength": int(payload_length),
        "alignment": int(alignment),
        "reserved": int(reserved),
    }


def read_trigsbin_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with Path(path).open("rb") as f:
        header = read_trigsbin_header(f)
        f.seek(header["manifestOffset"])
        manifest_bytes = f.read(header["manifestLength"])
    return header, json.loads(manifest_bytes.decode("utf-8"))


def inspect_trigsbin(args: argparse.Namespace) -> int:
    header, manifest = read_trigsbin_manifest(Path(args.input))
    print(json.dumps(header, indent=2))
    clips = manifest.get("clips", [])
    total_tensors = len(manifest.get("tensorIndex", []))
    if isinstance(manifest.get("static"), dict):
        total_static = int(manifest.get("static", {}).get("count", 0) or 0)
    else:
        total_static = sum(int(c.get("model", {}).get("staticCount", c.get("static", {}).get("count", 0)) or 0) for c in clips)
    total_dynamic = sum(int(c.get("model", {}).get("dynamicCount", c.get("dynamic", {}).get("count", 0)) or 0) for c in clips)
    print(
        f"format={manifest.get('format')} version={manifest.get('version')} "
        f"clips={len(clips)} tensors={total_tensors} staticSplats={total_static:,} dynamicSplats={total_dynamic:,}"
    )
    if args.manifest:
        print(json.dumps(manifest, indent=2))
    return 0


# -----------------------------------------------------------------------------
# CLI / summary
# -----------------------------------------------------------------------------

def print_summary(output: Path, manifest: dict[str, Any]) -> None:
    path = Path(output)
    size = path.stat().st_size if path.exists() else 0
    clips = manifest.get("clips", [])
    if isinstance(manifest.get("static"), dict):
        total_static = int(manifest.get("static", {}).get("count", 0) or 0)
    else:
        total_static = sum(int(c.get("model", {}).get("staticCount", c.get("static", {}).get("count", 0)) or 0) for c in clips)
    total_dynamic = sum(int(c.get("model", {}).get("dynamicCount", c.get("dynamic", {}).get("count", 0)) or 0) for c in clips)
    payload_length = int(manifest.get("container", {}).get("payloadLength", 0))
    payload_offset = int(manifest.get("container", {}).get("payloadOffset", 0))
    print(f"Wrote: {path}")
    print(f"  file size: {size:,} bytes")
    print(f"  payload offset: {payload_offset:,} bytes")
    print(f"  payload length: {payload_length:,} bytes")
    print(f"  clips: {len(clips)}")
    print(f"  static splats: {total_static:,}")
    print(f"  dynamic splats: {total_dynamic:,}")
    print(f"  tensors: {len(manifest.get('tensorIndex', []))}")


def add_common_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output", type=Path, required=True, help="Output .trigsbin runtime binary")
    p.add_argument("--alignment", type=int, default=DEFAULT_ALIGNMENT, help="Payload/tensor alignment in bytes. Default: 256")
    p.add_argument("--tensor-hashes", action="store_true", help="Add per-tensor SHA-256 hashes to the manifest")
    p.add_argument("--manifest-output", type=Path, default=None, help="Optional sidecar JSON manifest for debugging/engine integration")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build or inspect TRiGS runtime .trigsbin files")
    p.add_argument("--version", action="store_true", help="Print version and exit")
    sub = p.add_subparsers(dest="command")

    p_trigs = sub.add_parser("from-trigs", help="Convert an authoring .trigs archive into runtime .trigsbin")
    p_trigs.add_argument("--input", type=Path, required=True, help="Input .trigs authoring archive")
    add_common_output_args(p_trigs)
    p_trigs.set_defaults(func=convert_from_trigs)

    p_ckpt = sub.add_parser("from-checkpoints", help="Build runtime .trigsbin directly from trainer checkpoints")
    add_common_output_args(p_ckpt)
    p_ckpt.add_argument("--checkpoint", action="append", type=Path, default=[], help="Trainer checkpoint. Repeat once per clip, in timeline order.")
    p_ckpt.add_argument("--checkpoint-list", type=Path, default=None, help="Text file with one checkpoint path per line.")
    p_ckpt.add_argument(
        "--range",
        dest="ranges",
        action="append",
        default=[],
        help="Manual schedule: TRAIN_START:TRAIN_END:FULL_START:FULL_END. FULL is the full-opacity interval; fades occur outside it.",
    )
    p_ckpt.add_argument("--frame-rate", type=float, required=True, help="Timeline and training FPS, e.g. 60")
    p_ckpt.add_argument("--tensor-dtype", choices=SUPPORTED_DTYPES, default="motion32", help="Payload dtype policy. Default: motion32")
    p_ckpt.add_argument(
        "--static-export-mode",
        choices=("split", "all-dynamic"),
        default="split",
        help="split uses params['is_static']; all-dynamic preserves old single dynamic payload behavior.",
    )
    p_ckpt.add_argument(
        "--static-storage",
        choices=("global", "per-clip"),
        default="global",
        help=(
            "How to store static splats in from-checkpoints exports. "
            "global writes one top-level static group for the whole timeline; "
            "per-clip keeps the older behavior. Default: global."
        ),
    )
    p_ckpt.add_argument(
        "--static-source-clip",
        type=int,
        default=0,
        help="Clip index whose static rows become the global static group. Default: 0 / first clip.",
    )
    p_ckpt.add_argument(
        "--shared-static",
        choices=("off", "exact", "first"),
        default="off",
        help=(
            "Legacy per-clip static tensor reuse. Only used with --static-storage per-clip. "
            "Ignored when --static-storage global."
        ),
    )
    p_ckpt.add_argument(
        "--color-normalize",
        choices=("none", "dynamic-sh0-median", "dynamic-protect-sh0-median", "all-sh0-median"),
        default="none",
        help="Experimental bounded SH color normalization across clips. Default: none.",
    )
    p_ckpt.add_argument(
        "--color-normalize-reference",
        choices=("first", "previous"),
        default="previous",
        help="Reference anchor for color normalization. Default: previous clip.",
    )
    p_ckpt.add_argument(
        "--color-normalize-apply",
        choices=("sh0", "sh0-shN"),
        default="sh0",
        help="Which SH tensors to scale. sh0 is safer; sh0-shN is stronger. Default: sh0.",
    )
    p_ckpt.add_argument("--color-normalize-max-gain", type=float, default=1.12, help="Max per-channel gain clamp. Default: 1.12")
    p_ckpt.add_argument("--color-normalize-strength", type=float, default=1.0, help="Blend strength 0..1 for computed gain. Default: 1.0")
    p_ckpt.add_argument("--color-normalize-min-samples", type=int, default=1000, help="Minimum preferred-row samples before falling back. Default: 1000")
    p_ckpt.add_argument(
        "--overlap-color-calibrate",
        choices=(
            "none",
            "dynamic-sh0-median",
            "dynamic-protect-sh0-median",
            "all-sh0-median",
            "dynamic-shN-rms",
            "dynamic-protect-shN-rms",
            "all-shN-rms",
        ),
        default="none",
        help=(
            "Experimental boundary calibration using temporal overlap rows from adjacent clips. "
            "Computes a per-channel gain from previous/current overlap anchors and applies it to the current clip. "
            "shN-rms modes compare higher-order SH energy; sh0-median modes compare DC/base color. Default: none."
        ),
    )
    p_ckpt.add_argument(
        "--overlap-color-calibrate-apply",
        choices=("sh0", "sh0-shN"),
        default="sh0-shN",
        help="Which SH tensors to scale for overlap calibration. Default: sh0-shN.",
    )
    p_ckpt.add_argument("--overlap-color-calibrate-max-gain", type=float, default=1.08, help="Max per-channel gain clamp for overlap calibration. Default: 1.08")
    p_ckpt.add_argument("--overlap-color-calibrate-strength", type=float, default=1.0, help="Blend strength 0..1 for overlap calibration. Default: 1.0")
    p_ckpt.add_argument("--overlap-color-calibrate-min-samples", type=int, default=200, help="Minimum rows in overlap before falling back to whole-clip semantic group. Default: 200")
    p_ckpt.add_argument("--overlap-color-calibrate-min-alpha-percentile", type=float, default=25.0, help="Drop lowest-opacity overlap rows below this percentile before anchoring. Default: 25")
    p_ckpt.add_argument("--fade-frames", type=int, default=0, help="Symmetric opacity fade duration in frames at transitions")
    p_ckpt.add_argument("--fade-in-frames", type=int, default=None, help="Override fade-in frames. Defaults to --fade-frames")
    p_ckpt.add_argument("--fade-out-frames", type=int, default=None, help="Override fade-out frames. Defaults to --fade-frames")
    p_ckpt.add_argument("--fade-curve", choices=("smoothstep", "linear"), default="smoothstep")
    p_ckpt.add_argument("--transition-padding-frames", type=int, default=None, help="Defaults to max(fade-in, fade-out)")
    p_ckpt.add_argument("--preload-frames", type=int, default=60)
    p_ckpt.add_argument("--keep-alive-frames", type=int, default=60)
    p_ckpt.add_argument("--total-frames", type=int, default=None, help="Override timeline total frames. Defaults to max(core_end)")
    p_ckpt.add_argument("--loop", action="store_true")
    p_ckpt.add_argument("--name", type=str, default=None)
    p_ckpt.add_argument("--description", type=str, default=None)
    p_ckpt.set_defaults(func=convert_from_checkpoints)

    p_single = sub.add_parser(
        "from-checkpoint",
        help=(
            "Split one assembled long-sequence trainer checkpoint into "
            "temporally streamable runtime clips"
        ),
    )
    add_common_output_args(p_single)
    p_single.add_argument("--checkpoint", type=Path, required=True, help="Assembled long-sequence trainer checkpoint")
    p_single.add_argument("--frame-rate", type=float, required=True, help="Timeline and training FPS, e.g. 60")
    p_single.add_argument(
        "--chunk-duration-seconds",
        type=float,
        default=None,
        help="Override trainer windows with fixed-duration windows. By default, use checkpoint metrics.windows.",
    )
    p_single.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=0.5,
        help="Overlap for --chunk-duration-seconds. Ignored when trainer windows are used. Default: 0.5",
    )
    p_single.add_argument(
        "--max-chunk-duration-seconds",
        type=float,
        default=None,
        help=(
            "Split into the fewest equal fixed-duration windows no longer than this; a "
            "sequence up to this long becomes one clip. Exclusive with --chunk-duration-seconds."
        ),
    )
    p_single.add_argument(
        "--temporal-threshold",
        type=float,
        default=None,
        help=(
            "Temporal opacity support threshold. Defaults to the checkpoint trainer metric, "
            f"then the checkpoint's training config, then {TRAINER_DEFAULT_TEMPORAL_THRESHOLD} "
            "(the trainer default)."
        ),
    )
    p_single.add_argument(
        "--temporal-extra-margin-seconds",
        type=float,
        default=0.0,
        help="Extra support radius added on both sides of every dynamic row. Default: 0",
    )
    p_single.add_argument(
        "--time-domain",
        choices=("global", "clip-local"),
        default="clip-local",
        help=(
            "Time coordinates stored in sliced clips. clip-local matches the "
            "normal multi-checkpoint runtime format; global preserves source "
            "checkpoint time coordinates. Default: clip-local."
        ),
    )
    p_single.add_argument("--total-frames", type=int, default=None, help="Override total timeline frames")
    p_single.add_argument(
        "--allow-truncate",
        action="store_true",
        help="Allow --total-frames to cut off frames present in the trainer timeline",
    )
    p_single.add_argument("--tensor-dtype", choices=SUPPORTED_DTYPES, default="motion32", help="Payload dtype policy. Default: motion32")
    p_single.add_argument("--fade-frames", type=int, default=0, help="Symmetric opacity fade duration in frames at transitions")
    p_single.add_argument("--fade-in-frames", type=int, default=None, help="Override fade-in frames. Defaults to --fade-frames")
    p_single.add_argument("--fade-out-frames", type=int, default=None, help="Override fade-out frames. Defaults to --fade-frames")
    p_single.add_argument("--fade-curve", choices=("smoothstep", "linear"), default="smoothstep")
    p_single.add_argument("--transition-padding-frames", type=int, default=None, help="Defaults to max(fade-in, fade-out)")
    p_single.add_argument("--preload-frames", type=int, default=60)
    p_single.add_argument("--keep-alive-frames", type=int, default=60)
    p_single.add_argument("--loop", action="store_true")
    p_single.add_argument("--name", type=str, default=None)
    p_single.add_argument("--description", type=str, default=None)
    p_single.set_defaults(func=convert_from_single_checkpoint)

    p_inspect = sub.add_parser("inspect", help="Inspect a .trigsbin header and manifest summary")
    p_inspect.add_argument("--input", type=Path, required=True, help="Input .trigsbin")
    p_inspect.add_argument("--manifest", action="store_true", help="Print full embedded manifest JSON")
    p_inspect.set_defaults(func=inspect_trigsbin)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(VERSION)
        return 0
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
