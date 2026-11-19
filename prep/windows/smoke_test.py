#!/usr/bin/env python3
"""GPU smoke test for the Open4DGS Windows environment (imports + one call per CUDA extension)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ok = True


def check(name, fn):
    global ok
    t = time.time()
    try:
        info = fn()
        print(f"  OK   {name:28s} {time.time() - t:5.1f}s  {info or ''}")
    except Exception as exc:  # noqa: BLE001 - report every failure, keep going
        ok = False
        print(f"  FAIL {name:28s} {type(exc).__name__}: {exc}")


print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), python {sys.version.split()[0]}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
print(f"GPU: {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}")


def _knn():
    from pytorch3d.ops import knn_points
    x = torch.rand(1, 5000, 3, device="cuda")
    return tuple(knn_points(x, x, K=4).dists.shape)


def _gsplat():
    from gsplat import rasterization
    n = 20000
    means = torch.randn(n, 3, device="cuda") * 0.5
    means[:, 2] += 3
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device="cuda"), dim=-1)
    scales = torch.full((n, 3), 0.01, device="cuda")
    opac = torch.full((n,), 0.8, device="cuda")
    cols = torch.rand(n, 3, device="cuda", requires_grad=True)
    K = torch.tensor([[[500.0, 0, 320], [0, 500, 240], [0, 0, 1]]], device="cuda")
    img, _, _ = rasterization(means, quats, scales, opac, cols, torch.eye(4, device="cuda")[None], K, 640, 480)
    img.sum().backward()
    return tuple(img.shape)


def _local_corr():
    import romav2.local_correlation as lc
    if lc.local_corr is None:
        raise RuntimeError("fused-local-corr not importable; RoMa v2 'precise' will run out of memory on 8 GB GPUs")
    B, c, h, w, r = 1, 32, 24, 24, 2
    K = (2 * r + 1) ** 2
    f0, f1 = torch.randn(B, c, h, w, device="cuda"), torch.randn(B, c, h, w, device="cuda")
    warp = (torch.rand(B, h, w, 2, device="cuda") * 2 - 1) * 0.9
    g = torch.meshgrid([torch.linspace(-2 * r / h, 2 * r / h, 2 * r + 1, device="cuda"),
                        torch.linspace(-2 * r / w, 2 * r / w, 2 * r + 1, device="cuda")], indexing="ij")
    lw = torch.stack((g[1], g[0]), -1)[None].expand(1, 2 * r + 1, 2 * r + 1, 2).reshape(1, K, 2)
    ref = lc.native_torch_local_corr(f0, f1, warp, lw, B, K, c, r, h, w, "cuda")
    fus = lc.local_corr_wrapper(f0, f1, warp, lw, B, K, c, r, h, w, "cuda")
    err = (ref - fus).abs().max().item()
    if err > 1e-3:
        raise RuntimeError(f"fused kernel disagrees with reference (max abs diff {err:.2e})")
    return f"max abs diff vs reference {err:.1e}"


def _roma():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "init"))
    from roma_backends import create_roma_backend
    return create_roma_backend(backend="v2", device="cuda").name


def _memfof():
    from memfof import MEMFOF
    m = MEMFOF.from_pretrained("egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH").eval().to("cuda")
    with torch.inference_mode():
        flow = m(torch.zeros((1, 3, 3, 256, 256), device="cuda"))["flow"][-1]
    return tuple(flow.shape)


def _trigs():
    import trigs  # noqa: F401
    from trigs.cli import train_from_colmap  # noqa: F401
    import pycolmap
    return f"pycolmap {pycolmap.__version__}"


check("pytorch3d knn_points", _knn)
check("gsplat rasterization+backward", _gsplat)
check("fused-local-corr", _local_corr)
check("RoMa v2 model (weights)", _roma)
check("MEMFOF model (weights)", _memfof)
check("trigs trainer import", _trigs)
raise SystemExit(0 if ok else 1)
