# Azure Kinect import

These tools convert multi-device Azure Kinect DK recordings (`k4arecorder` `.mkv`,
wired master/subordinate sync) into the layout the Open4DGS pipeline reads:
`images/`, `sparse/0`, `reference_export.json` and `sync.json`. You do not need
the Azure Kinect Sensor SDK. The tools read the MKV files with `ffmpeg` and `ffprobe`.

| Step | Script | Produces |
|---|---|---|
| 1. Audit | `prep/extract/extract_azure_kinect.py inspect` | `k4a_report/inspect.json`, `timing.json`, `sync_sheet.jpg`, `depth_overlay.jpg` |
| 2. Extract | `prep/extract/extract_azure_kinect.py extract` | `images/`, `depth/`, `k4a_cameras.json`, `sync.json`, `frame_map.json` |
| 3. Calibrate | `prep/align/calibrate_azure_kinect.py` | `sparse/0`, `reference_export.json`, `k4a_report/calibration*.{json,jpg,ply}` |

All steps can be run again. Step 2 reuses the timing solution saved by step 1;
pass `--resolve-timing` to compute it again.

## Requirements

- `ffmpeg` and `ffprobe` on `PATH`.
- Python 3.10 or newer with `numpy`, `opencv-python` and `scipy`. Install
  `pycolmap` too if you want to read back the model. The prep environment from
  [prep/SETUP.md](../SETUP.md) already has all of these.
- Recordings must contain the COLOR and DEPTH tracks. Calibration uses depth for
  metric scale.

## Run

Put one capture's `.mkv` files (one per device) in a folder. The sorted filenames
set the camera folder order `0001`, `0002`, and so on. The device recorded as
`MASTER` defines logical time.

```bash
CAP=datasets/professor           # folder holding the .mkv files
OUT=$CAP/o4d                     # capture data directory for the pipeline

python prep/extract/extract_azure_kinect.py inspect --input-dir $CAP --output-dir $OUT --rotate l
# review the QA files listed below, then:
python prep/extract/extract_azure_kinect.py extract --input-dir $CAP --output-dir $OUT --rotate l
python prep/align/calibrate_azure_kinect.py --data-dir $OUT
```

`--rotate` turns the images upright. The choices are `n` (none), `l` (90° counter-clockwise), `r`
(90° clockwise) and `u` (180°). Use `--rotate-camera NAME=ROT` to set one device
differently. The intrinsics, the depth→colour transform and the poses are all
rotated with the images. Check `sync_sheet.jpg`: people should stand upright.
Upright images matter for the RoMa/DINO matchers used later.

Useful options:

| Option | Use |
|---|---|
| `--start-frame N --num-frames M` | Export a sub-window of the common frame range. |
| `--manual-lag NAME=L` | Force the integer frame lag for a device whose clock cannot be measured. |
| `--undistort-alpha 1` | Keep the full undistorted field of view, including black borders. The default `0` crops to valid pixels. |
| `--no-depth` | Skip depth export. Calibration needs depth, so only use this when you already have `sparse/0`. |
| calibrate `--num-sample-frames` | Number of synchronised instants used for matching. The default is 8. Raise it for low-texture scenes. |
| calibrate `--world-frame reference` | Keep the master camera as the world origin instead of the floor frame. |

## What the tools do

**Calibration conversion.** Each MKV embeds the factory `calibration.json`. Its
intrinsics are normalised to the full sensor: 4096×3072 for colour and 1024×1024
for depth. `k4a_mkv.mode_intrinsics` applies the SDK's mode crop and scale; for
example, 1080p crops 384 rows top and bottom, then scales by 1920/4096. It maps the
Brown-Conrady parameters to OpenCV's rational model. The SDK stores `p2` before
`p1`.

**Frame alignment** (`k4a_sync.py`):

1. *Device clocks.* Devices whose colour timestamps share one time base form a
   clock group. Within that group the frame correspondence and sub-frame offset
   come directly from the timestamps. Daisy-chain delays (160 µs steps) show up
   here.
2. *Visual sync.* Devices on another clock are aligned to the reference group by
   cross-correlating a view-independent motion signal: the fraction of pixels that
   change between frames. For subordinate devices, `K4A_SUBORDINATE_DELAY_NS`
   fixes the sub-frame phase. The visual estimate only has to choose the integer
   frame.
3. *Validation.* The visual estimator also runs on every timestamp-measured
   device. The console prints its worst error against that ground truth. If it
   reports `CHECK`, do not trust visual sync for this capture; use
   `--manual-lag`.

Slots are frame numbers derived from timestamps, so a dropped frame leaves a hole
instead of shifting later frames. Extraction uses the longest logical range that
every camera covers, after skipping each device's first two packets, which are
lock-in transients.

**Extrinsics** (`calibrate_azure_kinect.py`):

1. Extract RootSIFT features on undistorted frames at several synchronised
   instants.
2. Lift each keypoint to 3D using that camera's own depth.
3. Estimate each camera pair with PnP-RANSAC.
4. Initialise poses with a maximum spanning tree from the master camera.
5. Run a bundle adjustment over poses and points: pixel reprojection plus a weak
   ToF depth prior, with a Huber loss and two rounds of outlier rejection.
6. Set the world frame. y = 0 is the RANSAC floor plane from fused depth, +Y is up
   (the trainer viewer's convention), and the origin is the floor point below the
   spot the cameras look at. Units are metres.

## QA checklist

| File | Pass if |
|---|---|
| console: `visual-sync validation` | Max error is below about 0.3 frames. |
| console: `[visual+subordinate_delay]` notes | No `WARNING ... from nearest physical hypothesis`. |
| `k4a_report/sync_sheet.jpg` | Poses match across cameras on each row and people are upright. |
| `k4a_report/depth_overlay.jpg` | Depth edges (checkerboard tiles) line up with colour edges, e.g. the person and light stands. |
| `k4a_report/inspect.json` → `timestamps.*.dropped_frames` | 0, or at least not inside the window you need. |
| calibrate console: per-camera reprojection | Median below 1 px. |
| calibrate console: `depth vs others` | Median about 10–25 mm. ToF bias between devices makes 0 unreachable. |
| calibrate console: camera heights | They match the physical rig. |
| `k4a_report/calibration_reprojection.jpg` | Coloured dots from the other cameras land on the matching surfaces in every view. |
| `k4a_report/calibration_fused.ply` | Opens in MeshLab or CloudCompare as one coherent scene, with no doubled surfaces. |

Reference result for `datasets/professor` (6 devices, 1080p MJPEG, NFOV depth,
30 fps):

- Visual sync validation error: 0.12 frames.
- `sub-04` and `sub-05` run on a separate clock with an 18.1–18.3 ms subordinate
  delay. They resolve to lag +1 with a −15.2 / −15.0 ms offset.
- Common window: 377 frames (12.6 s).
- Bundle adjustment: 0.40 px median reprojection, 4 mm median depth residual.
- Cameras sit on a ring of about 1.45 m radius, 1.74–1.80 m above the floor.

## Per-camera depth offsets

Each Kinect's ToF depth carries its own bias. On the calibrated rig, neighbouring
cameras disagree about the depth of the same surface by a constant offset of up to 15 mm.
`prep/extract/k4a_depth_offsets.py` measures that disagreement on the subject and
solves for one additive correction per camera. Run it after `k4a_register_depth.py`
and the masks:

```bash
python "$REPO/prep/extract/k4a_depth_offsets.py" --data-dir .   # -> depth_offsets.json
```

Inputs are the generic registered-depth contract (`depth_registered/`, z-depth of the
colour camera), `reference_export.json` and a mask directory (`--masks-dir`, default
`masks_refined`). Nothing in it is Kinect-specific.

**Method.** The script samples `--num-frames` (12) frames spread over the clip. For
every frame and ordered camera pair (i, j) it takes camera i's depth inside its mask
eroded by 4 px, away from depth edges (7×7 depth range ≤ 1.5%) and on surfaces that face
both cameras within 60° (`--max-view-angle`; this drops far-side matches). It lifts
those pixels to 3D, projects them into j and takes the residual z_j(point) − d_j
(bilinear, edge-safe, inside j's mask). It gates the residuals twice, then takes the
median. Each pair's frame medians are combined by their median over frames. With
corrections o the residual becomes r + c·o_i − o_j, where c = (n·ray_i)/(n·ray_j) for
surface normal n. It is *not* the z-component of ray i in j, because j's lookup slides
along the surface. A least-squares fit over the pairs gives o. `--huber-mm` turns on
IRLS.

**Gauge.** A common shift of all cameras only enters through c − 1. That is small,
since the overlapping surfaces face both cameras, and it looks the same as a rig-scale
error. The fit includes the shift and `--gauge mean` (the default) removes it: the
corrections sum to zero and keep the depth level that the calibration scale was fitted
to. `--gauge free` keeps it, but it is not reliable. For an independent check the
script also compares colour-triangulated points (`--gauge-points`, default
`k4a_report/calibration_points.npy`, or a COLMAP `points3D.txt`) with the measured depth
at their projections. That check is reported, never applied. `sparse/0` from
`calibrate_azure_kinect.py` has no 3D points.

**Output contract** (`depth_offsets.json`):

| Key | Content |
|---|---|
| `contract` | `"metres to ADD to every valid depth value of the camera (z along the colour optical axis)"` |
| `offsets_m` | `{camera folder: metres}`. Add it to valid depth only; 0 stays invalid. |
| `gauge` | `mean` or `free` |
| `frames` | the frames sampled |
| `pair_residual_rms_mm` | `{before, after}` over the pair equations |
| `per_pair` | per ordered pair `"0001->0002"`: points, frames, c, residual before and after, frame medians |
| `per_frame_offsets_mm`, `per_frame_spread_mm` | the same solve on each frame alone, and its std per camera |
| `free_solution`, `gauge_points` | the common-shift diagnostics described above |

Consumers: `prep/init/depth_init.py --depth-offsets-json`. The trainer's depth loss
implements the same contract.

**Measured on `datasets/professor`** (12 frames over all 377, `masks_refined`, 12 s on
CPU). Twelve neighbour pairs were measured; opposite cameras never pass the facing test.

- Offsets to add: 0001 −7.4, 0002 −9.5, 0003 −0.6, 0004 +4.7, 0005 +6.3, 0006 +6.5 mm.
- Pair residual RMS: 9.0 → 4.1 mm. The largest pair goes from 15.4 to 4.2 mm.
- Stable over time: per-frame solves have a std of 0.4–0.8 mm and stay within ±1.6 mm of
  the result over the whole clip. Changing the erosion, the facing angle (45°), the masks,
  IRLS or the frames (1–90 only) moves the offsets by ≤ 1.5 mm. At 75°, three grazing
  pairs are admitted (RMS 5.9 mm) and the offsets shrink by up to 6 mm.
- The fit leaves 4.1 mm RMS. The residual has a lateral gradient within each pair (up
  to ±10 mm across the subject), which a per-camera z offset cannot model. The likely
  cause is calibration or depth→colour registration error.
- Common shift: not identifiable. The free solution gives −6.4 mm (conditioning 0.07),
  but it varies from −0.3 to −13 mm between settings. The calibration points (1,989,
  mostly background, median depth 1.8–2.6 m) give −5.8 mm. Per camera, though, they
  differ from the subject offsets by −11.8 to +2.9 mm (std 5 mm), so the ToF bias likely
  depends on distance and material. Hence the default
  `--gauge mean`.

## Feeding the pipeline

On Windows, `prep/windows/run_kinect_pipeline.ps1` runs every step from the MKV
files to the exported 4DGS. See [prep/windows/README.md](../windows/README.md).
The equivalent manual commands follow.

Run these from `$OUT`. The capture is 30 fps and genlocked, so skip the visual
sync solver (pipeline stage 4). Per-camera offsets are already in `sync.json`.

```bash
python "$REPO/prep/mask/depth_masks.py" --data-dir .
# review mask_report/mask_sheet.jpg, then:

python "$REPO/prep/init/prep-roma-trigs-memory.py" --reference-json reference_export.json \
  --images-root images --fps 30 --sync-json sync.json \
  --frame-start 1 --frame-end <LAST> --masks-root masks --output-path keyframes_masked.npz

trigs-train-colmap --data-dir . --output-dir runs/<name> --npz-path keyframes_masked.npz \
  --masks-root masks --config.masked-rgb-loss --config.n-steps 30000
```

The mask step writes `masks/<cam>/<frame:06d>.png` from the Kinect depth (255 =
subject); see [prep/mask/README-depth-masks.md](../mask/README-depth-masks.md). With
`--masks-root masks`, prep-roma does not initialise background points. The trainer
needs `--config.masked-rgb-loss` as well; `--masks-root` alone only adds masked PSNR
metrics. For an unmasked run, drop the mask step and both `--masks-root` and
`--config.masked-rgb-loss`, and write `keyframes.npz` instead, so the two
initialisations do not overwrite each other.

The trainer reads `<data-dir>/sync.json` automatically. Do not pass
`--timing-json` as well. Do not create a `frame_timestamps.json`: the trainer
would prefer it over `sync.json`.

## Caveats

- **Few views.** Six cameras about 60° apart is far sparser than the 27 and 45
  camera example captures. Expect weaker novel views between cameras and more
  floaters. Masks help.
- **Background.** Walls and the floor are visible to every camera. Without masks,
  prep-roma initialises the floor and anything else within about 2.2 m of the centre
  (the limit comes from `reference_export.json`'s `scene_meta`, built on the camera
  ring of about 1.5 m radius), and training fits it. Use the depth masks
  ([prep/mask/README-depth-masks.md](../mask/README-depth-masks.md); the `mask` stage
  of the Windows runner, on by default) so that only the subject is initialised and
  opacity outside it is trained to 0. Otherwise edit the result in SuperSplat.
- **Rolling shutter.** The Kinect colour camera is rolling shutter and its readout
  time is not modelled. This matters little for slow motion. For fast motion,
  consider `--config.learn-camera-time-offsets` in the trainer.
- **Depth is not used for initialisation yet.** `depth/` and `k4a_cameras.json`
  hold everything needed to build a depth-based initial NPZ instead of using RoMa.
  This is a possible next step.
