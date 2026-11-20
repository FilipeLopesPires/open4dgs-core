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
| `run_kinect_pipeline.ps1` | Takes an Azure Kinect capture to a trained 4DGS model in one command: inspect, extract, calibrate, mask, init, train, export (plus refine, register and foregrounds when the quality options need them). |

## Setup

Prerequisites:

- An NVIDIA driver at version 570 or newer.
- Git.
- Visual Studio 2022 with **Desktop development with C++**. CUDA 12.8 rejects the
  VS 2026 (MSVC 14.5x) toolset, but both versions can be installed side by side.

Optional: `winget` for uv; the script installs uv itself when it is missing.

Put the checkouts side by side, as `open4dgs-core-fork` (or `open4dgs-core`) and
`open4dgs-trigs-fork` (or `open4dgs-trigs`). Then run:

```powershell
powershell -ExecutionPolicy Bypass -File prep\windows\setup_env.ps1 -TrigsRepo ..\open4dgs-trigs-fork
```

Without `-TrigsRepo` the script uses the sibling `open4dgs-trigs` checkout. The
runner's `-DepthLoss` option and the viewer's `--up` option need the trainer from
`open4dgs-trigs-fork`, which adds depth supervision (`--depth-root`,
`--config.lambda-depth`) and those viewer options.

To switch an existing environment to another trainer checkout without a full setup:

```bat
call C:\o4d\o4d-env.cmd
uv pip install --python C:\o4d\venv\Scripts\python.exe --no-deps -e <path>\open4dgs-trigs-fork
```

Then change the `OPEN4DGS_TRIGS` line in `C:\o4d\o4d-env.cmd` to the same path.

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
| `refine` (with `-MasksDir masks_refined`) | `prep\mask\refine_masks.py` | `mask_report\refine.json` (not one from a `--qa-only` run) |
| `register` (with `-DepthLoss`) | `prep\extract\k4a_register_depth.py` | `depth_registered\_complete.json` |
| `foregrounds` (with `-Composite`) | `prep\mask\straight_foregrounds.py` | `foregrounds[_refined]\_complete.json`, if newer than the mask report and covering the frame range |
| `init` | `prep-roma-trigs-memory.py`, or `prep\init\depth_init.py` with `-Init depth` | the keyframe file, see below |
| `train` | `trigs-train-colmap` | `runs\<Name>\trigs_ckpt.pt` |
| `export` | `trigs-export-plys` + `make-trigs-timeline.py` | `runs\<Name>\<Name>.trigsbin` |

`refine`, `register` and `foregrounds` are added automatically when the option that
needs them is given and their output is missing; you do not list them in `-Stages`.

The keyframe file is `keyframes[_depth][_masked|_refined]_f<start>-<end>.npz`: `_depth`
with `-Init depth`, `_masked` with the default masks, `_refined` with
`-MasksDir masks_refined`, neither with `-NoMasks`. Each combination keeps its own file.

Each stage writes `<capture>\o4d\logs\<stage>.log`; `init`, `train` and `export` write
`init-<keyframe file>.log`, `train-<Name>.log` and `export-<Name>.log`, so runs do not
overwrite each other's logs. A stage is skipped when its output already exists. Pass `-Force` to redo a stage. `-Stages train,export -Name run2 -Steps 45000`
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

### Quality options

With only six cameras, colour alone leaves the depth of many splats ambiguous: a model
can reproduce every camera image and still fall apart between cameras (doubled faces,
needle-shaped splats, floor smears). These options use the Kinect depth and cleaner
masks to remove that ambiguity. All are off by default.

| Parameter | Effect |
|---|---|
| `-MasksDir masks_refined` | Train and initialise with the colour-refined masks (`refine` stage, [README-refine-masks.md](../mask/README-refine-masks.md)). The depth masks keep a safety margin with floor and wall in it; the refined masks follow the subject's edge. Needs SAM 2. |
| `-Init depth` | Initialise from the registered Kinect depth instead of RoMa v2 + MEMFOF ([README-depth-init.md](../init/README-depth-init.md)). Dense, metric points on the whole visible surface; about 1 min instead of about 20 min for 90 frames. |
| `-DepthInitPoints <n>` | Points per keyframe with `-Init depth` (default 60000). Also sets `--config.target-count-per-keyframe`, which otherwise subsamples every keyframe to 25,862 points. More points cost file size; see the result below. |
| `-DepthLoss <w>` | Supervises the rendered depth with the registered Kinect depth: `--depth-root depth_registered --config.lambda-depth <w>`. `1.0` is the tested value. Needs the trainer from `open4dgs-trigs-fork`. |
| `-Composite` | Mattes the ground truth on a white plate and supervises the whole frame (`--config.masked-rgb-composite`), so a splat outside the subject costs RGB loss, not only an alpha penalty. Builds the straight foreground images first. |
| `-MaxAnisotropy <r>` | Caps the ratio of a splat's largest to smallest axis (`--config.max-scale-anisotropy`). `10` removes the needle-shaped splats that streak in novel views. |
| `-TrainExtra "<args>"` | Appended verbatim to the `trigs-train-colmap` command. |
| `-TrainerSrc <dir>` | Runs `train` and `export` with `PYTHONPATH=<dir>`, to try another trainer checkout without reinstalling it. |

Best tested recipe, 3 s of the `professor` capture:

```powershell
powershell -ExecutionPolicy Bypass -File prep\windows\run_kinect_pipeline.ps1 `
    -Capture datasets\professor -Rotate l -Stages init,train,export -Name all3s `
    -FrameEnd 90 -Steps 15000 -ChunkSeconds 0 -MasksDir masks_refined -Init depth `
    -DepthLoss 1.0 -Composite -MaxAnisotropy 10 -AlphaOutsideWeight 1.0
```

Result on an RTX 3070 (about 2.5 h of training, 3.8 GB of GPU memory):

- Views between the cameras are coherent: one face instead of a doubled one, sharp
  limbs and laptop, no floor smears. Remaining defects: spiky hair at the back of the
  head, thin wisps along the silhouette, and a partly missing chair (the refined masks
  cut most of it away).
- Rendered depth against the Kinect depth (trainer eval): median error 4.4 mm,
  90th percentile 20 mm.
- The trainer's masked eval PSNR is 26.8 dB. A run without the depth options scored
  35.6 dB but broke in the views between cameras. By default no camera is held out
  (`--config.holdout-views` is empty), so this metric scores the training cameras and
  rewards overfitting them. It is also not comparable across modes: with `-Composite`
  the eval uses a binarised mask while training uses the soft mask. Judge runs by
  novel views, or hold a camera out for A/B tests.
- 1.08 M splats; the `.trigsbin` is 353 MB for 3 s (24 MB for the run without depth
  init). Reducing the size is open work.

Measured on an RTX 3070 with the `professor` capture (6 cameras, 1080×1920, 377 frames):

| Stage | Time | Notes |
|---|---|---|
| inspect + extract + calibrate | about 2.5 min | |
| mask | 83 s | CPU only. The subject covers about 9–10% of each image. |
| init (RoMa v2 precise + MEMFOF) | about 7 min per 31 frames | About 1.5 h for 377 frames. |
| train | about 0.3 s/step at 31 frames | The 30k-step default is several hours. `-ChunkSeconds 3` (the default) trains 3 s windows so the full sequence fits in 8 GB. |
| export | under 1 min | Per-frame PLYs plus a `.trigsbin` for the SuperSplat TRiGS editor. |

### View the result

```bat
call C:\o4d\o4d-env.cmd
python -m trigs.core.viewer --ckpt <out>\runs\<Name>\trigs_ckpt.pt --port 8080 --fps 30 ^
    --total-frames <frames> --temporal-threshold 0.01 --up +y --no-spatial-filter
```

Open `http://localhost:8080`. To watch training live, replace `--ckpt ...` with
`--watch <out>\runs\<Name>\viewer_snapshots`. Run one viewer at a time, or give each its
own `--port`.

The calibration puts the floor at y = 0 with +Y up; without `--up +y` the viewer orbits
around viser's default +Z axis and the subject appears lying on its side. Add
`--no-spatial-filter` for a seated or standing subject: the viewer's default filter hides
splats beyond 1.1x the 95th-percentile distance from the splat centre, which removes the
shoe soles (the points farthest from the centre).
