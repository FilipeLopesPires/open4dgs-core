"""CPU fidelity check of an .asog made from an Open4DGS run against the run's per-frame PLYs.

Decodes the .asog with the animated-sog reference decoder (asog.decode, the players' twin), evaluates it at
frame k / fps, turns it back to the trainer's y-up frame (undo the encoder's --transform z180) and compares
with plys/frame_<k>.ply (the checkpoint evaluated per frame by trigs' export_plys):

- visible splat counts and the effective opacity mass (opacity x temporal weight);
- nearest-neighbour distance of every opaque .asog splat to the PLY splats (orientation / motion errors
  show up as centimetres here);
- height range (floor at y = 0, +Y up);
- a CPU point render (alpha-weighted DC colour per pixel, orthographic front and side views) of both,
  their PSNR, and a side-by-side PNG.

No GPU. Usage (inside C:\\o4d\\venv, PYTHONPATH=<animated-sog-fork>\\python\\src):
    python asog_fidelity.py <clip.asog> <run>\\plys <out_dir> [--frames 0 45 89] [--fps 30]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from asog import decode

SH_C0 = 0.28209479177387814


def read_ply(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        names, n = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        a = np.frombuffer(f.read(n * 4 * len(names)), np.float32).reshape(n, len(names))
    col = {k: i for i, k in enumerate(names)}
    return dict(xyz=a[:, [col["x"], col["y"], col["z"]]].astype(np.float64),
                sh0=a[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]].astype(np.float64),
                opacity=1 / (1 + np.exp(-a[:, col["opacity"]].astype(np.float64))))


def point_render(xyz, rgb, alpha, axes, lo, hi, px=0.004, blur=1.0):
    """Orthographic splat-as-point render: image axes `axes` = (horizontal, vertical) world axes."""
    w, h = (np.ceil((hi - lo) / px).astype(int) + 1)[list(axes)]
    u = np.clip(((xyz[:, axes[0]] - lo[axes[0]]) / px).astype(int), 0, w - 1)
    v = np.clip(h - 1 - ((xyz[:, axes[1]] - lo[axes[1]]) / px).astype(int), 0, h - 1)
    idx = v * w + u
    a = np.clip(alpha, 0, 0.999)
    acc = np.stack([np.bincount(idx, a * rgb[:, c], w * h) for c in range(3)], 1)
    wsum = np.bincount(idx, a, w * h)
    trans = np.bincount(idx, np.log1p(-a), w * h)
    col = acc / np.maximum(wsum, 1e-9)[:, None]
    img = col * (1 - np.exp(trans))[:, None]
    img = img.reshape(h, w, 3)
    return np.stack([gaussian_filter(img[..., c], blur) for c in range(3)], -1).clip(0, 1)


def psnr(a, b):
    return float(10 * np.log10(1 / max(np.mean((a - b) ** 2), 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("asog", type=Path)
    ap.add_argument("plys", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 45, 89])
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    clip = decode.read(args.asog)
    man = clip.manifest
    z180 = man.get("source", {}).get("transform") == "z180"
    back = np.array([-1.0, -1.0, 1.0]) if z180 else np.array([1.0, 1.0, 1.0])
    report = dict(asog=str(args.asog), count=clip.count, threshold=man["motion"].get("threshold", 0.05),
                  sm=man["motion"]["sm"], source=man.get("source"), frames={})
    rows = []
    for k in args.frames:
        t = (k + 0.0) / args.fps
        fr = clip.evaluate(t)
        vis = fr.weights > 0
        a_xyz = fr.positions[vis] * back
        a_alpha = clip.opacity[vis] * fr.weights[vis]
        a_rgb = (0.5 + SH_C0 * clip.sh0[vis]).clip(0, 1)
        ply = read_ply(args.plys / f"frame_{k:06d}.ply")
        p_rgb = (0.5 + SH_C0 * ply["sh0"]).clip(0, 1)
        tree = cKDTree(ply["xyz"])
        opaque = a_alpha > 0.1
        d, _ = tree.query(a_xyz[opaque], k=1)
        lo = np.percentile(ply["xyz"], 0.5, axis=0) - 0.05
        hi = np.percentile(ply["xyz"], 99.5, axis=0) + 0.05
        imgs = {}
        for view, axes in (("front", (0, 1)), ("side", (2, 1))):
            ia = point_render(a_xyz, a_rgb, a_alpha, axes, lo, hi)
            ip = point_render(ply["xyz"], p_rgb, ply["opacity"], axes, lo, hi)
            imgs[view] = (ip, ia, psnr(ia, ip))
        r = dict(t=t, asog_visible=int(vis.sum()), ply_count=len(ply["xyz"]),
                 asog_opacity_mass=float(a_alpha.sum()), ply_opacity_mass=float(ply["opacity"].sum()),
                 nn_mm_median=float(np.median(d) * 1000), nn_mm_p90=float(np.percentile(d, 90) * 1000),
                 nn_mm_p99=float(np.percentile(d, 99) * 1000),
                 asog_y_p1_p99=[float(x) for x in np.percentile(a_xyz[opaque][:, 1], [1, 99])],
                 ply_y_p1_p99=[float(x) for x in np.percentile(ply["xyz"][:, 1], [1, 99])],
                 psnr_front=imgs["front"][2], psnr_side=imgs["side"][2])
        report["frames"][k] = r
        print(json.dumps({k: r}))
        row = np.concatenate([np.concatenate([imgs[v][0], imgs[v][1]], 1) for v in ("front", "side")], 1)
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows], 0)
    Image.fromarray((sheet * 255).astype(np.uint8)).save(args.out / "fidelity_ply_vs_asog.png")
    (args.out / "fidelity.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out / 'fidelity_ply_vs_asog.png'} (per row: front PLY | front asog | side PLY | side asog)")


if __name__ == "__main__":
    main()
