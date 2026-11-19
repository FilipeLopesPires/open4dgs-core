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
| `run_kinect_pipeline.ps1` | Takes an Azure Kinect capture to a trained 4DGS model in one command: inspect, extract, calibrate, mask, init, train, export. |

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

Stages, in order. All paths are under `<capture>\o4d` (`-Out` to change):

| Stage | Runs | Skipped when this exists |
|---|---|---|
| `inspect` | `extract_azure_kinect.py inspect` | `k4a_report\inspect.json` |
| `extract` | `extract_azure_kinect.py extract` | `k4a_cameras.json` |
| `calibrate` | `calibrate_azure_kinect.py` | `reference_export.json` |
| `mask` | `prep\mask\depth_masks.py` | `mask_report\masks.json` (not one from a `--qa-only` run) |
| `init` | `prep-roma-trigs-memory.py` | `keyframes_masked_f<start>-<end>.npz`, or `keyframes_f<start>-<end>.npz` with `-NoMasks` |
| `train` | `trigs-train-colmap` | `runs\<Name>\trigs_ckpt.pt` |
| `export` | `trigs-export-plys` + `make-trigs-timeline.py` | `runs\<Name>\<Name>.trigsbin` |

Each stage writes `<capture>\o4d\logs\<stage>.log` and is skipped when its output
already exists. Pass `-Force` to redo a stage. `-Stages train,export -Name run2 -Steps 45000`
re-trains under a new run name. Check the QA images listed in
[README-azure-kinect.md](../extract/README-azure-kinect.md) after `calibrate` and before
the long stages. Use `-Stages inspect,extract,calibrate` to stop there, or
`-Stages inspect,extract,calibrate,mask` to also check the masks.

### Masks

Masks are on by default. The `mask` stage writes `masks\<cam>\<frame>.png` (255 =
subject) from the Kinect depth; see
[prep/mask/README-depth-masks.md](../mask/README-depth-masks.md). Check
`mask_report\mask_sheet.jpg` before `init`. With masks:

- `init` adds `--masks-root masks`: background (floor, walls) is not initialised. The
  output is `keyframes_masked_f<start>-<end>.npz`, e.g. `keyframes_masked_f0001-0377.npz`.
  The frame range is part of the name, so a short test (`-FrameEnd 90`) and a full-clip
  run each keep their own initialisation instead of one silently reusing the other's.
- `train` adds `--masks-root masks --config.masked-rgb-loss`: outside the mask there is
  no RGB loss and opacity is pushed to 0.

| Parameter | Effect |
|---|---|
| `-NoMasks` | Skip the `mask` stage; `init` and `train` run without masks and use `keyframes_f<start>-<end>.npz` (the behaviour before masks were added). |
| `-AlphaOutsideWeight <w>` | Adds `--config.lambda-alpha-outside-mask <w>` (trainer default 0.1). |
| `-AlphaMaskWeight <w>` | Adds `--config.lambda-alpha-mask <w>` (trainer default 0.5). |

The alpha weights are passed only when given. The trainer README's masked recipe uses
`-AlphaOutsideWeight 1.0 -AlphaMaskWeight 0.05`; it has not been tuned for Kinect data.

Masked and unmasked runs share the run folder `runs\<Name>` (default `full`), and
`train` is skipped when that folder already has a checkpoint. To keep both, give each
run its own name, for example `-NoMasks -Name nomask`.

Measured on an RTX 3070 with the `professor` capture (6 cameras, 1080×1920, 377 frames):

| Stage | Time | Notes |
|---|---|---|
| inspect + extract + calibrate | about 2.5 min | |
| mask | 83 s | CPU only. The subject covers about 9–10% of each image. |
| init (RoMa v2 precise + MEMFOF) | about 7 min per 31 frames | About 1.5 h for 377 frames. |
| train | about 0.3 s/step at 31 frames | The 30k-step default is several hours. `-ChunkSeconds 3` (the default) trains 3 s windows so the full sequence fits in 8 GB. |
| export | under 1 min | Per-frame PLYs plus a `.trigsbin` for the SuperSplat TRiGS editor. |

To watch training, run `python -m trigs.core.viewer --watch <out>\runs\<name>\viewer_snapshots --port 8080`.
