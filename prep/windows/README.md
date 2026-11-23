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
    -Capture datasets\<capture> -Rotate l
```

This processes the whole capture with the [default recipe](#default-recipe--preset-best) and
ends with `runs\full\full.asog` for the three.js player ([Play in the browser](#play-in-the-browser)).
Training takes about 0.6 s per step, i.e. about 100 s per frame: about 10 h for the
377-frame test capture on an RTX 3070.

The measurements in this file come from one test capture: 6 Azure Kinect DK devices,
1080×1920 portrait colour, NFOV depth, 377 frames at 30 fps, recorded in 2021 in the capture
stage described at <https://github.com/FilipeLopesPires/HoloStudio#capture-studio>.

Stages, in order. All paths are under `<capture>\o4d` (`-Out` to change):

| Stage | Runs | Skipped when this exists |
|---|---|---|
| `inspect` | `extract_azure_kinect.py inspect` | `k4a_report\inspect.json` |
| `extract` | `extract_azure_kinect.py extract` | `k4a_cameras.json` |
| `calibrate` | `calibrate_azure_kinect.py` | `reference_export.json` |
| `mask` | `prep\mask\depth_masks.py` | `mask_report\masks.json` (not one from a `--qa-only` run) |
| `refine` (with `-MasksDir masks_refined`) | `prep\mask\refine_masks.py` | `mask_report\refine.json` (not one from a `--qa-only` run) |
| `consistent` (with `-MasksDir masks_consistent`) | `prep\mask\consistent_masks.py --unknown` for the frame range | `mask_report\consistent.json` covering the frame range, newer than `refine.json` |
| `offsets` (with `-DepthOffsets`) | `prep\extract\k4a_depth_offsets.py` | `depth_offsets.json` |
| `register` (with `-DepthLoss`) | `prep\extract\k4a_register_depth.py` | `depth_registered\_complete.json` |
| `foregrounds` (with `-Composite`) | `prep\mask\straight_foregrounds.py` | `foregrounds[_refined]\_complete.json`, if newer than the mask report and covering the frame range |
| `init` | `prep-roma-trigs-memory.py`, or `prep\init\depth_init.py` with `-Init depth` | the keyframe file, see below |
| `train` | `trigs-train-colmap` | `runs\<Name>\trigs_ckpt.pt` |
| `export` | `deliver\prepare_export_checkpoint.py` + `trigs-export-plys` + `make-trigs-timeline.py` (see [Export and file size](#export-and-file-size)) | `runs\<Name>\<Name>.trigsbin` |
| `asog` | `asog flipbook flip` + `make-trigs-timeline.py` + `asog encode` (see [Play in the browser](#play-in-the-browser)) | `runs\<Name>\<Name>.asog` |

`refine`, `consistent`, `offsets`, `register` and `foregrounds` are added automatically when the option that
needs them is given and their output is missing; you do not list them in `-Stages`.

The keyframe file is `keyframes[_depth][_masked|_refined|_consistent][_doff]_f<start>-<end>.npz`:
`_depth` with `-Init depth`, `_masked` with the default masks, `_refined` with
`-MasksDir masks_refined`, `_consistent` with `-MasksDir masks_consistent`, neither with
`-NoMasks`; `_doff` with `-Init depth -DepthOffsets`. Each combination keeps its own file.

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

Further options (need the trainer from `open4dgs-trigs-fork`; all off by default):

| Parameter | Effect |
|---|---|
| `-HoldoutViews <i>[,<j>]` | Leaves cameras out of training (0-based position in the sorted camera folders: `3` = `0004`) and scores them as `psnr/holdout_*`, `depth/holdout_*` in TensorBoard. The only metric that measures views the model was not fitted to; use it for every A/B test. |
| `-MasksDir masks_consistent` | Cross-view vote over the refined masks (`consistent` stage, `prep\mask\consistent_masks.py --unknown` for the frame range, then `foregrounds_consistent`): restores mask misses that at least two other cameras' depth supports, and marks pixels the cameras disagree on as unknown (mask PNG alpha), which the trainer leaves out of its mask losses. See [README-refine-masks.md](../mask/README-refine-masks.md). |
| `-DepthOffsets` | Per-camera Kinect depth corrections (`offsets` stage, `prep\extract\k4a_depth_offsets.py` → `depth_offsets.json`), applied to `-Init depth` (keyframe file `..._doff_...`) and to the `-DepthLoss` targets. See [README-azure-kinect.md](../extract/README-azure-kinect.md). |
| `-DepthEdgePx <px>` | `--config.depth-edge-px` (trainer default 1): depth pixels this close to a hole or depth jump are not supervised. `2` halves the background-contaminated depth pixels along the silhouette for 6% fewer supervised pixels. |
| `-DepthLossFinal <s>` | Keeps `-DepthLoss` until step `-DepthDecayStart` (5000), then lowers it log-linearly to `-DepthLoss × s` over `-DepthDecaySteps` (2000). Suggested `0.15`. |
| `-LearnAppearance` | `--config.learn-camera-appearance`: a colour gain and bias per camera. The cameras' exposures differ by up to ±10% (`0005` +11%, `0006` −8%); without it the splat colours average them. |
| `-RandomBackground` | `--config.masked-rgb-random-bg` (with `-Composite`): a random plate colour per rendered camera, so the white plate cannot be learned into the edge splats. |
| `-MaxAnisotropy 4` | Candidate instead of 10: with 10, the 99th percentile of the splats' long/middle axis ratio sits at the cap (full 3 s run). Not yet checked on thin parts (fingers, laptop lid edge-on) over a full run. |
| `-ShDegree <0-3>` | `--sh-degree` (default 3). Lower degrees give six cameras less view-dependent colour to overfit. |
| `-DepthDistortion <w>` | `--config.lambda-depth-distortion` from step `-DepthDistortionStart`: pulls the splats a pixel sees towards one depth. |
| `-OpacityBinary <w>` | `--config.lambda-opacity-binary` from step `-OpacityBinaryStart`: pushes opacities to 0 or 1 (late in training). |
| `-LearnTimeOffsets` | `--config.learn-camera-time-offsets` (from step 8000; `-TrainExtra "--config.camera-time-offset-warmup-steps <n>"` to change). The rig is hardware-synced, so expect offsets near 0. Do not set `--config.rs-readout-ms` without a measured readout. |

Smoke tests of these options (frames 1–31, 1500 steps, camera `0004` held out, on top of
the 3 s recipe below; metrics at step 1000, so only large differences mean anything):

| Run | Held-out masked / interior PSNR | Held-out depth median / p90 | Notes |
|---|---|---|---|
| recipe | 16.5 / 18.8 dB | 16.4 / 63 mm | median opacity 0.084, 2.7% above 0.5 |
| `-DepthDistortion 100 -DepthDistortionStart 300` | 15.4 / 17.4 dB | 9.6 / 35 mm | geometry much tighter in the held-out view, colour −1 dB: try 10–30 |
| + `-OpacityBinary 0.05 -OpacityBinaryStart 750 -MaxAnisotropy 4` | 15.0 / 16.9 dB | 9.0 / 36 mm | median opacity 1.0 (90% above 0.5): far too strong, try 0.005 late |
| `-DepthOffsets -DepthEdgePx 2 -DepthLossFinal 0.15` (decay 600–1000) | 16.1 / 18.1 dB | 14.8 / 61 mm | |
| `-LearnAppearance -RandomBackground` | 16.4 / 18.8 dB | 13.6 / 44 mm | learned gains did not reproduce the measured exposure ratios in 1500 steps |
| `-MasksDir masks_consistent -ShDegree 1 -LearnTimeOffsets` | 17.1 / 18.8 dB | 16.2 / 64 mm | masked PSNR leaves unknown pixels out; one offset reached the 8.3 ms clamp |

### Default recipe (`-Preset best`)

The runner applies this recipe unless an option is given on the command line (`-Preset basic`
restores the trainer defaults):

```
-MasksDir masks_refined -Init depth -DepthLoss 1.0 -Composite -AlphaOutsideWeight 1.0
-MaxAnisotropy 10 -DepthOffsets -DepthEdgePx 2   -Steps = 165 x frames (rounded to 1000)
```

It was chosen with A/B runs on frames 1–31 (5000 steps, camera `0004` held out; metrics at
step 4000):

| Run | Held-out depth median / p90 | Held-out PSNR full / interior | Training-camera interior PSNR |
|---|---|---|---|
| qA0: 3 s recipe below | 16.9 / 44.6 mm | 22.41 / 19.25 dB | 39.71 dB |
| **qA1: + `-DepthOffsets -DepthEdgePx 2`** | **13.7 / 39.5 mm** | **22.28 / 19.43 dB** | **39.81 dB** |
| qA2: qA1 + `-DepthDistortion 20` (from 1000) + `-OpacityBinary 0.005` (from 3500) | 11.5 / 33.3 mm | 21.93 / 18.73 dB | 38.31 dB |

- qA1 makes the geometry about 20% more accurate in the held-out view at the same colour.
- The depth-distortion term (qA2) tightens the geometry further but costs about 0.5 dB of
  colour in the held-out view and 1.4 dB on the training cameras, so it stays opt-in.
- qA1 and qA2 also passed `-DepthLossFinal 0.15`, but with the default `-DepthDecayStart
  5000` the decay never started in a 5000-step run. Constant depth loss is what was tested,
  so the recipe leaves `-DepthLossFinal` off.
- Novel views between the cameras look nearly the same for all three. The views next to the
  held-out camera are slightly cleaner with qA2.

Earlier recipe, 3 s of the test capture:

```powershell
powershell -ExecutionPolicy Bypass -File prep\windows\run_kinect_pipeline.ps1 `
    -Capture datasets\<capture> -Rotate l -Stages init,train,export -Name all3s `
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
  rewards overfitting them. That trainer also binarised the mask in the composite eval
  while training used the soft mask, which capped a perfect fit at about 30 dB (fixed
  since; the trainer also logs `psnr/eval_interior`). Judge runs by novel views, or hold
  a camera out (`-HoldoutViews`) for A/B tests.
- 1.08 M splats; the `.trigsbin` was 353 MB for 3 s with the old export and is 178 MB
  with the current defaults (see [Export and file size](#export-and-file-size)).

Measured on an RTX 3070 with the test capture (6 cameras, 1080×1920, 377 frames):

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

### Play in the browser

The `asog` stage converts the run to Animated SOG with
[animated-sog](https://github.com/arrival-space/animated-sog): about 9x smaller than the
`.trigsbin` (the 3 s test clip: 178 MB → 20.5 MB), with full spherical harmonics.

The export stage's `.trigsbin` is +Y up and holds the trainer's wxyz quaternions, which
`make-trigs-timeline.py` labels `xyzw`; the players expect y down. So the stage takes
animated-sog's own route: `asog flipbook flip` turns a fresh export copy 180° about z and
writes xyzw quaternions, `make-trigs-timeline.py` packs it to `runs\<Name>\<Name>_ydown.trigsbin`,
and `asog encode` turns that into `runs\<Name>\<Name>.asog` with no extra options.

- The encoder is the `asog` package of the environment, or a checkout: `-AsogRepo <dir>`, else
  `animated-sog-fork` or `animated-sog` next to this repo when present.
- Encoder options go through `-AsogArgs`, e.g. `-AsogArgs '--prune','0'` keeps the 30% of the
  dynamic splats the encoder drops by default.
- The SH palette k-means runs on the GPU when there is one (minutes for 3 M splats).
  `-AsogDevice cpu` hides CUDA, e.g. while another job trains; without CUDA the stock encoder
  limits the palette size.
- The players fade splats with a temporal threshold of 0.05; the runs train with 0.01, so short-lived
  splats (fast hands) fade slightly earlier in the player than in training.

Serve the player and open the URL the stage prints:

```powershell
powershell -ExecutionPolicy Bypass -File prep\windows\asog_convert.ps1 -Serve
# http://localhost:8765/animated-sog/examples/three/?src=/open4dgs-core/datasets/<capture>/o4d/runs/<Name>/<Name>.asog
```

The server shares the folder that holds both repositories; three.js comes from a CDN.
`asog_convert.ps1 -Trigsbin <run>\<Name>_ydown.trigsbin -Out <file>.asog -EncodeArgs ...` re-encodes
with other encoder options. `asog_fidelity.py` compares an `.asog` with the run's PLYs, and
`asog_screenshots.mjs` takes headless screenshots of the player.

### Export and file size

The `export` stage never reads or changes `trigs_ckpt.pt` beyond loading it. It runs:

1. `deliver\prepare_export_checkpoint.py` writes `runs\<Name>\export_ckpt.pt`, a copy that
   - carries the run's training config. `trigs_ckpt.pt` has none, so both exporters used
     to fall back to defaults; the copy takes it from the final viewer snapshot
     (`viewer_snapshots\latest.json`). The frame count and fps come from the same place,
     so `-FrameEnd` is not needed for an export-only run.
   - leaves out rows with base opacity below `-PruneOpacity` (default 0.005; 0 keeps all).
2. `trigs-export-plys` writes `runs\<Name>\plys\` from the copy (skipped with `-NoPly`).
   With `-CompressPly`, `deliver\compress_plys.py` then replaces each `frame_<n>.ply` by
   `frame_<n>.compressed.ply`.
3. `make-trigs-timeline.py from-checkpoint` writes `runs\<Name>\<Name>.trigsbin` from the
   copy, then the copy is deleted (`export_ckpt.pt.json` records what was resolved).

| Parameter | Default | Effect |
|---|---|---|
| `-FromRun <run>` | | Export `runs\<run>\trigs_ckpt.pt` into `runs\<Name>`, without training. |
| `-PruneOpacity <a>` | 0.005 | Rows with `sigmoid(opacity) < a` are not exported. |
| `-TensorDtype <policy>` | `mixed` | `.trigsbin` dtypes. `mixed`: fp16 except `times` and `log_durations`. `motion32` (the old default) also keeps positions, rotations and motion in fp32. |
| `-PackChunkSeconds <s>` | 10 | Longest `.trigsbin` clip. A sequence up to `s` long is one clip; a longer one is split into the fewest equal clips no longer than `s`. |
| `-NoPly` | | No per-frame PLYs. |
| `-CompressPly` | | PLYs in the PlayCanvas compressed format. SuperSplat reads it; tools that read only the standard 3DGS PLY layout do not. |

Example: re-export the `all3s` run with the current settings, without retraining:

```powershell
powershell -ExecutionPolicy Bypass -File prep\windows\run_kinect_pipeline.ps1 `
    -Capture datasets\<capture> -Stages export -FromRun all3s -Name all3s_compressed -CompressPly
```

**Temporal threshold.** Each row's opacity fades with a Gaussian in time, renormalised so it
reaches zero at the threshold `tau` (`--config.temporal-active-threshold`, trainer default
0.01). The packager used to fall back to 0.05 when the checkpoint recorded nothing, so the
`.trigsbin` was packaged for a different fade than the one trained. It now takes the
threshold from the export copy's config (falling back to the trainer default 0.01), and the
PLY export uses the same config. The TRiGS editor does not read the threshold from the file:
it renders with its own **Temporal opacity threshold** slider in the view options, default
0.05. Set it to 0.010 to see the model as trained.

**Clips.** A row is stored in every clip whose time range its temporal support touches. On
the 3 s run, a clip boundary in the middle of the sequence is crossed by 0.42 M of the
1.08 M rows, about 1.2 s worth of rows (0.36 M rows/s). So each boundary costs about
1.2 s / chunk length: +47% at 2.5 s chunks, +23% at 5 s, +12% at 10 s. The old fixed 2.5 s
chunks split the 3 s run into 2 clips (0 to 2.5 s, 2.5 to 3 s) and stored 31% of the rows
twice. Longer clips keep more rows resident at once in the player; lower
`-PackChunkSeconds` if the editor runs out of memory on a long capture.

Measured on the 3 s run of the test capture (`all3s`, 1,078,242 rows):

| | `.trigsbin` | Rows stored | Bytes per row |
|---|---|---|---|
| Old export (2.5 s chunks, `motion32`, threshold 0.05) | 353,273,248 B | 1,412,874 | 250 |
| Threshold fixed only (2.5 s chunks, `motion32`) | 366,864,883 B | 1,467,233 | 250 |
| Current defaults (one clip, prune 0.005, `mixed`) | 178,010,350 B | 977,902 | 182 |

The per-frame PLYs went from 6.78 GB (old export) to 1.74 GB with `-PruneOpacity 0.005`
and `-CompressPly` (6.69 GB without `-CompressPly`).

Quality, rendered from the 6 training cameras and 6 views halfway between neighbouring
cameras at frames 0, 37, 74, 75, 76 and 89 (PSNR on the subject; the `.trigsbin` decoded
from its tensors, tau from the file):

| Comparison | Training views | Views between cameras |
|---|---|---|
| Current defaults vs threshold fixed only (compression) | 55.9 dB (min 52.7) | 55.0 dB (min 49.5) |
| Threshold fixed vs old export (the fix) | 46.9 dB (min 32.4) | 45.8 dB (min 34.4) |
| Against the masked ground truth: trained model / old export / fixed / current | 31.27 / 30.81 / 31.27 / 31.26 dB | |

Pruning removes 9.3% of the rows. fp16 rounds a coordinate below 2 m to within 0.5 mm.
Compressed PLYs score 45 dB against the uncompressed ones.
