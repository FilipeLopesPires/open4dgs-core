#!/usr/bin/env python3
"""Azure Kinect (k4arecorder) MKV helpers: metadata, calibration, timestamps, decoding.

The Azure Kinect recorder writes Matroska files containing:

* ``COLOR``  track - MJPEG (or NV12/YUY2/BGRA) colour frames
* ``DEPTH``  track - ``b16g`` 16-bit big-endian depth in millimetres
* ``IR``     track - ``b16g`` 16-bit active-IR intensity
* ``IMU``    track
* ``calibration.json`` attachment - the raw *factory* calibration

Matroska block timestamps are ``device_timestamp - K4A_START_OFFSET_NS``.
Devices in one wired-sync chain whose clocks were reset together therefore share
a device time base, which this module exposes as ``device_time_ns``.

The factory calibration stores intrinsics normalised to the full sensor
(4096x3072 colour, 1024x1024 depth).  ``mode_intrinsics`` reproduces the Sensor
SDK's mode-specific conversion (crop + scale) so the result matches what
``k4a_device_get_calibration`` would return for the recorded resolution.

Only ``ffmpeg``/``ffprobe`` executables plus NumPy/OpenCV are required; the Azure
Kinect Sensor SDK is *not* needed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Mode tables (Azure Kinect Sensor SDK, src/transformation/mode_specific_calibration.c)
# calibration_resolution, crop_offset, output_resolution
# --------------------------------------------------------------------------------------
_COLOR_MODES = {
    "720P": ((4096, 3072), (0, 384), (1280, 720)),
    "1080P": ((4096, 3072), (0, 384), (1920, 1080)),
    "1440P": ((4096, 3072), (0, 384), (2560, 1440)),
    "2160P": ((4096, 3072), (0, 384), (3840, 2160)),
    "1536P": ((4096, 3072), (0, 0), (2048, 1536)),
    "3072P": ((4096, 3072), (0, 0), (4096, 3072)),
}
_DEPTH_MODES = {
    "NFOV_2X2BINNED": ((512, 512), (96, 90), (320, 288)),
    "NFOV_UNBINNED": ((1024, 1024), (192, 180), (640, 576)),
    "WFOV_2X2BINNED": ((512, 512), (0, 0), (512, 512)),
    "WFOV_UNBINNED": ((1024, 1024), (0, 0), (1024, 1024)),
    "PASSIVE_IR": ((1024, 1024), (0, 0), (1024, 1024)),
}


@dataclass
class Intrinsics:
    """Pinhole + OpenCV rational distortion (k1,k2,p1,p2,k3,k4,k5,k6)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray  # shape (8,)
    metric_radius: float = 0.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "dist_opencv_rational": [float(v) for v in self.dist],
            "metric_radius": self.metric_radius,
        }


@dataclass
class K4ARecording:
    path: Path
    name: str
    serial: str
    sync_mode: str  # MASTER / SUBORDINATE / STANDALONE
    subordinate_delay_ns: int
    depth_delay_ns: int
    start_offset_ns: int
    color_mode: str  # e.g. MJPG_1080P
    depth_mode: str  # e.g. NFOV_UNBINNED
    color_stream: int
    depth_stream: int | None
    ir_stream: int | None
    color_size: tuple[int, int]
    depth_size: tuple[int, int] | None
    fps: float
    raw_calibration: dict = field(repr=False)
    color_pts_ns: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    depth_pts_ns: np.ndarray | None = field(default=None, repr=False)

    # ----------------------------------------------------------------------------------
    @property
    def color_device_ns(self) -> np.ndarray:
        return self.color_pts_ns + self.start_offset_ns

    @property
    def depth_device_ns(self) -> np.ndarray | None:
        if self.depth_pts_ns is None:
            return None
        return self.depth_pts_ns + self.start_offset_ns

    @property
    def color_resolution(self) -> str:
        return self.color_mode.split("_", 1)[1]

    def color_intrinsics(self) -> Intrinsics:
        cam = _find_camera(self.raw_calibration, "PhotoVideo")
        intr = mode_intrinsics(cam, _COLOR_MODES[self.color_resolution])
        if (intr.width, intr.height) != self.color_size:
            raise ValueError(
                f"{self.name}: colour mode {self.color_mode} gives {intr.width}x{intr.height} "
                f"but the stream is {self.color_size[0]}x{self.color_size[1]}"
            )
        return intr

    def depth_intrinsics(self) -> Intrinsics:
        cam = _find_camera(self.raw_calibration, "Depth")
        intr = mode_intrinsics(cam, _DEPTH_MODES[self.depth_mode])
        if self.depth_size is not None and (intr.width, intr.height) != self.depth_size:
            raise ValueError(f"{self.name}: depth mode {self.depth_mode} size mismatch")
        return intr

    def depth_to_color(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (R, t_m) mapping depth-camera points to colour-camera points: Xc = R Xd + t."""
        depth = _find_camera(self.raw_calibration, "Depth")
        color = _find_camera(self.raw_calibration, "PhotoVideo")
        Rd, td = _rt(depth)
        Rc, tc = _rt(color)
        # Each camera's Rt maps the device reference frame (depth camera D0, identity) to that camera.
        R = Rc @ Rd.T
        t = tc - R @ td
        return R, t


def _rt(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(cam["Rt"]["Rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(cam["Rt"]["Translation"], dtype=np.float64).reshape(3)
    return R, t


def _find_camera(raw: dict, purpose: str) -> dict:
    for cam in raw["CalibrationInformation"]["Cameras"]:
        if cam["Purpose"].endswith(purpose):
            return cam
    raise KeyError(f"calibration has no camera with purpose {purpose}")


def mode_intrinsics(cam: dict, mode) -> Intrinsics:
    """Convert normalised factory intrinsics to a recording mode (SDK-equivalent)."""
    (cw, ch), (ox, oy), (ow, oh) = mode
    p = cam["Intrinsics"]["ModelParameters"]
    if not cam["Intrinsics"]["ModelType"].endswith("BrownConrady"):
        raise ValueError(f"unsupported lens model {cam['Intrinsics']['ModelType']}")
    cx, cy, fx, fy, k1, k2, k3, k4, k5, k6, codx, cody, p2, p1 = p[:14]
    if abs(codx) > 1e-9 or abs(cody) > 1e-9:
        raise ValueError("non-zero centre of distortion (codx/cody) is not representable in OpenCV")
    # normalised -> calibration-resolution pixels (corner origin), crop, scale to output.
    # The scale follows the horizontal crop; depth NFOV's vertical crop is not centred.
    scale = ow / float(cw - 2 * ox)
    fx_o = fx * cw * scale
    fy_o = fy * ch * scale
    cx_o = (cx * cw - ox) * scale - 0.5  # -> OpenCV pixel-centre convention
    cy_o = (cy * ch - oy) * scale - 0.5
    dist = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)
    return Intrinsics(
        width=ow,
        height=oh,
        fx=float(fx_o),
        fy=float(fy_o),
        cx=float(cx_o),
        cy=float(cy_o),
        dist=dist,
        metric_radius=float(cam.get("MetricRadius", 0.0)),
    )


# --------------------------------------------------------------------------------------
# ffprobe helpers
# --------------------------------------------------------------------------------------
def _require(tool: str) -> str:
    exe = shutil.which(tool)
    if exe is None:
        raise SystemExit(f"{tool} not found on PATH")
    return exe


def _ffprobe_json(path: Path, *args: str) -> dict:
    cmd = [_require("ffprobe"), "-v", "error", "-of", "json", *args, str(path)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def _packet_pts_ns(path: Path, stream: int) -> np.ndarray:
    cmd = [
        _require("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        str(stream),
        "-show_entries",
        "packet=pts_time",
        "-of",
        "csv=p=0",
        str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    vals = [line.strip().rstrip(",") for line in out.splitlines() if line.strip()]
    # Matroska stores microsecond-ish precision; round to the nearest ns-free microsecond.
    return np.round(np.array([float(v) for v in vals]) * 1e9).astype(np.int64)


def _read_attachment(path: Path, stream_index: int) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "calibration.json"
        cmd = [_require("ffmpeg"), "-v", "error", "-y", f"-dump_attachment:{stream_index}", str(out), "-i", str(path)]
        # ffmpeg exits non-zero ("At least one output file must be specified") after dumping.
        subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
        if not out.exists():
            raise RuntimeError(f"could not extract calibration attachment from {path}")
        return out.read_bytes()


def open_recording(path: Path, name: str | None = None, read_timestamps: bool = True) -> K4ARecording:
    path = Path(path).resolve()
    info = _ffprobe_json(path, "-show_format", "-show_streams")
    tags = {k.upper(): v for k, v in info["format"].get("tags", {}).items()}
    color = depth = ir = None
    calib_idx = None
    for s in info["streams"]:
        st = {k.upper(): v for k, v in s.get("tags", {}).items()}
        title = st.get("TITLE", "").upper()
        if s["codec_type"] == "video" and title == "COLOR":
            color = s
        elif s["codec_type"] == "video" and title == "DEPTH":
            depth = s
        elif s["codec_type"] == "video" and title == "IR":
            ir = s
        elif s["codec_type"] == "attachment" and st.get("FILENAME", "") == "calibration.json":
            calib_idx = s["index"]
    if color is None:
        raise ValueError(f"{path}: no COLOR track")
    if calib_idx is None:
        raise ValueError(f"{path}: no calibration.json attachment")
    raw_cal = json.loads(_read_attachment(path, calib_idx).decode("utf-8"))
    ctags = {k.upper(): v for k, v in color.get("tags", {}).items()}
    dtags = {k.upper(): v for k, v in (depth or {}).get("tags", {}).items()}
    num, den = (int(v) for v in color["r_frame_rate"].split("/"))
    rec = K4ARecording(
        path=path,
        name=name or path.stem,
        serial=tags.get("K4A_DEVICE_SERIAL_NUMBER", ""),
        sync_mode=tags.get("K4A_WIRED_SYNC_MODE", "STANDALONE"),
        subordinate_delay_ns=int(tags.get("K4A_SUBORDINATE_DELAY_NS", 0)),
        depth_delay_ns=int(tags.get("K4A_DEPTH_DELAY_NS", 0)),
        start_offset_ns=int(tags.get("K4A_START_OFFSET_NS", 0)),
        color_mode=ctags.get("K4A_COLOR_MODE", f"UNKNOWN_{color['height']}P"),
        depth_mode=dtags.get("K4A_DEPTH_MODE", "OFF"),
        color_stream=int(color["index"]),
        depth_stream=int(depth["index"]) if depth else None,
        ir_stream=int(ir["index"]) if ir else None,
        color_size=(int(color["width"]), int(color["height"])),
        depth_size=(int(depth["width"]), int(depth["height"])) if depth else None,
        fps=num / den,
        raw_calibration=raw_cal,
    )
    if read_timestamps:
        rec.color_pts_ns = _packet_pts_ns(path, rec.color_stream)
        if rec.depth_stream is not None:
            rec.depth_pts_ns = _packet_pts_ns(path, rec.depth_stream)
    return rec


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------
def iter_frames(
    path: Path,
    stream: int,
    width: int,
    height: int,
    pix_fmt: str,
    first: int = 0,
    count: int | None = None,
    scale: tuple[int, int] | None = None,
):
    """Yield (index, ndarray) decoded frames of one stream, in packet order, no duplication.

    ``pix_fmt`` is an ffmpeg raw format: ``bgr24``, ``gray`` or ``gray16be`` (depth/IR).
    Frames are returned in *decode order index*, which equals the packet index because
    k4arecorder writes intra-only tracks.
    """
    ow, oh = scale if scale else (width, height)
    channels = {"bgr24": 3, "gray": 1, "gray16be": 1}[pix_fmt]
    dtype = np.dtype(">u2") if pix_fmt == "gray16be" else np.dtype(np.uint8)
    frame_bytes = ow * oh * channels * dtype.itemsize
    vf = [f"select=gte(n\\,{first})"] if first else []
    if scale:
        vf.append(f"scale={ow}:{oh}:flags=area")
    cmd = [_require("ffmpeg"), "-v", "error", "-i", str(path), "-map", f"0:{stream}", "-fps_mode", "passthrough"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if count is not None:
        cmd += ["-frames:v", str(count)]
    cmd += ["-pix_fmt", pix_fmt, "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 2)
    idx = first
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, dtype=dtype)
            shape = (oh, ow, channels) if channels == 3 else (oh, ow)
            arr = arr.reshape(shape)
            if dtype.itemsize == 2:
                arr = arr.astype(np.uint16)
            yield idx, arr
            idx += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.kill()
        proc.wait()


def read_frame(rec: K4ARecording, kind: str, index: int) -> np.ndarray:
    """Decode a single frame (``kind`` in color/depth/ir) by packet index."""
    if kind == "color":
        w, h = rec.color_size
        gen = iter_frames(rec.path, rec.color_stream, w, h, "bgr24", first=index, count=1)
    else:
        stream = rec.depth_stream if kind == "depth" else rec.ir_stream
        if stream is None or rec.depth_size is None:
            raise ValueError(f"{rec.name}: no {kind} track")
        w, h = rec.depth_size
        gen = iter_frames(rec.path, stream, w, h, "gray16be", first=index, count=1)
    for _, frame in gen:
        return frame
    raise IndexError(f"{rec.name}: {kind} frame {index} not found")
