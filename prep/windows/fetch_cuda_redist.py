#!/usr/bin/env python3
"""Install a user-local CUDA toolkit on Windows from NVIDIA's redistributable archives.

No administrator rights, no system-wide installer, and no display-driver changes.
Downloads the per-component ``windows-x86_64`` zip archives listed in NVIDIA's
``redistrib_<version>.json`` manifest, verifies each SHA-256, and merges them into one
toolkit tree (``bin``, ``include``, ``lib\\x64``, ``nvvm``) that PyTorch's
``torch.utils.cpp_extension`` accepts as ``CUDA_HOME``.

Example::

    python -I prep/windows/fetch_cuda_redist.py --version 12.8.1 \
        --download-dir C:\\o4d\\cuda-dl --dest C:\\o4d\\cuda-12.8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"

# Everything PyTorch CUDA extensions (gsplat, PyTorch3D, fused-local-corr) need to
# compile and link: nvcc + CRT, runtime, CCCL (thrust/cub), and the library headers
# that ATen's CUDA headers include (cuBLAS, cuSPARSE, cuSOLVER, cuRAND, cuFFT).
COMPONENTS = [
    "cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_nvrtc", "cuda_nvtx", "cuda_profiler_api", "cuda_cupti",
    "libcublas", "libcusparse", "libcusolver", "libcurand", "libcufft", "libnvjitlink", "libnvfatbin",
    "cuda_cuobjdump", "cuda_nvdisasm", "cuda_nvprune", "cuda_cuxxfilt", "cuda_nvml_dev", "visual_studio_integration",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="12.8.1", help="CUDA redist manifest version (match the PyTorch wheel)")
    ap.add_argument("--download-dir", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path)
    args = ap.parse_args(argv)

    args.download_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.download_dir / f"redistrib_{args.version}.json"
    if not manifest_path.exists():
        urllib.request.urlretrieve(f"{BASE}redistrib_{args.version}.json", manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    args.dest.mkdir(parents=True, exist_ok=True)
    for name in COMPONENTS:
        entry = manifest.get(name, {}).get("windows-x86_64")
        if entry is None:
            print(f"skip {name}: not published for windows-x86_64 in {args.version}")
            continue
        archive = args.download_dir / Path(entry["relative_path"]).name
        if not archive.exists() or sha256(archive) != entry["sha256"]:
            print(f"download {name} ({int(entry['size']) / 2**20:.0f} MiB)", flush=True)
            urllib.request.urlretrieve(BASE + entry["relative_path"], archive)
        if sha256(archive) != entry["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {archive}")
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                parts = Path(info.filename).parts
                if info.is_dir() or len(parts) < 2:
                    continue
                rel = Path(*parts[1:])  # drop the archive's top-level folder
                if rel.is_absolute() or ".." in rel.parts:
                    raise SystemExit(f"unsafe path in {archive}: {info.filename}")
                out = args.dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        print(f"ok {name} {manifest[name]['version']}", flush=True)

    nvcc = args.dest / "bin" / "nvcc.exe"
    if not nvcc.exists():
        raise SystemExit("nvcc.exe missing after extraction")
    print(f"CUDA toolkit ready: {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
