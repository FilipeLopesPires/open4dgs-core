# Windows setup and Azure Kinect runner

These scripts set up preparation and training on Windows 11 with an NVIDIA GPU, without
administrator rights. They were tested on an RTX 3070 8 GB with NVIDIA driver 610.88 and
Visual Studio 2022 17.14 (MSVC 14.44).

| File | Purpose |
|---|---|
| `setup_env.ps1` | Installs everything. It can be run again; finished steps are skipped. Ends with GPU smoke tests. |
| `fetch_cuda_redist.py` | Installs a user-local CUDA toolkit from NVIDIA's per-component redistributable archives, with SHA-256 checks. No system installer and no driver change. |
| `fused-local-corr-msvc.patch` | Lets RoMa v2's CUDA correlation kernel build with MSVC. Upstream only ships Linux wheels. |
| `smoke_test.py` | Runs one CUDA call through each native extension and loads the RoMa v2 and MEMFOF weights. |
| `run_kinect_pipeline.ps1` | Takes an Azure Kinect capture to a trained 4DGS model in one command: inspect, extract, calibrate, init, train, export. |

## Setup

Prerequisites:

- An NVIDIA driver at version 570 or newer.
- Git.
- Visual Studio 2022 with **Desktop development with C++**. CUDA 12.8 rejects the
  VS 2026 (MSVC 14.5x) toolset, but both versions can be installed side by side.

Optional: `winget` for uv; the script installs uv itself when it is missing.

Put the checkouts side by side, as `open4dgs-core-fork` (or `open4dgs-core`) and
`open4dgs-trigs`. Then run:

```powershell
powershell -ExecutionPolicy Bypass -File prep\windows\setup_env.ps1
```

This creates the following under `C:\o4d`. Use `-Root` to choose another folder.
Keep it outside OneDrive or Nextcloud: the environment is about 8 GB of small files,
and sync clients lock files during builds.

| Path | Contents |
|---|---|
| `venv\` | Python 3.11 with PyTorch 2.10.0+cu128, the TriGS trainer, gsplat 1.5.3 and PyTorch3D 0.7.9 (built for this GPU), RoMa v2, MEMFOF and fused-local-corr |
| `cuda-12.8\` | CUDA 12.8.1 compiler and libraries, used only to build extensions |
| `o4d-env.cmd` | Environment script |

To use the environment, open `cmd.exe` and run:

```bat
call C:\o4d\o4d-env.cmd
```

It loads the VS 2022 compiler, CUDA, the venv, `REPO` and `OPEN4DGS_TRIGS`.

How this differs from the Linux guide, and why:

- **CUDA comes from redistributables, not the system installer.** The installer
  needs admin rights and bundles an older display driver.
- **fused-local-corr is built and patched.** Without it, RoMa v2 uses a pure-PyTorch
  fallback. At the default `precise` setting that fallback needs more than 8 GB
  (out of memory), while the fused kernel peaks at about 3 GB. The patch changes no
  maths: it swaps headers and namespace aliases (`torch/extension.h` breaks under
  MSVC with `'std': ambiguous symbol`) and sets MSVC compiler flags. The smoke test
  checks the kernel against RoMa's own reference implementation.
- **`uv sync --inexact`.** It keeps the preparation backends that are not in the
  trainer lockfile.

## Process an Azure Kinect capture

```powershell
powershell -ExecutionPolicy Bypass -File prep\windows\run_kinect_pipeline.ps1 `
    -Capture datasets\professor -Rotate l
```

Each stage writes `<capture>\o4d\logs\<stage>.log` and is skipped when its output
already exists. Pass `-Force` to redo a stage. `-Stages train,export -Name run2 -Steps 45000`
re-trains under a new run name. Check the QA images listed in
[README-azure-kinect.md](../extract/README-azure-kinect.md) after `calibrate` and before
the long stages. Use `-Stages inspect,extract,calibrate` to stop there.

Measured on an RTX 3070 with the `professor` capture (6 cameras, 1080×1920, 377 frames):

| Stage | Time | Notes |
|---|---|---|
| inspect + extract + calibrate | about 2.5 min | |
| init (RoMa v2 precise + MEMFOF) | about 7 min per 31 frames | About 1.5 h for 377 frames. |
| train | about 0.3 s/step at 31 frames | The 30k-step default is several hours. `-ChunkSeconds 3` (the default) trains 3 s windows so the full sequence fits in 8 GB. |
| export | under 1 min | Per-frame PLYs plus a `.trigsbin` for the SuperSplat TRiGS editor. |

To watch training, run `python -m trigs.core.viewer --watch <out>\runs\<name>\viewer_snapshots --port 8080`.
