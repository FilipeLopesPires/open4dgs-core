# Foreground masks from depth

`prep/mask/depth_masks.py` builds per-camera foreground masks from multi-view metric
depth. It works with any rig that records depth alongside colour. It needs no
background plate, no segmentation network and no GPU.

The masks feed point initialisation (`prep/init/prep-roma-trigs-memory.py`) and
training (`trigs-train-colmap`); see [How the masks are used](#how-the-masks-are-used).
For RGB-only rigs (no depth), use `prep/mask/generate_masks.py` instead (Background
Matting V2 with a clean plate, or SAM 2).

On Windows, `prep/windows/run_kinect_pipeline.ps1` runs it as the `mask` stage, between
`calibrate` and `init`.

## Run

From the capture data directory (the folder with `reference_export.json` and `images/`):

```bash
python "$REPO/prep/mask/depth_masks.py" --data-dir .
```

By default every frame common to all cameras in `images/` is processed. Use
`--frame-start` / `--frame-end` (inclusive) for a sub-range.

Outputs:

| Path | Contents |
|---|---|
| `masks/<cam>/<frame:06d>.png` | 8-bit alpha mask, same size as the image. 255 = subject, 0 = background, intermediate values on the feathered edge. |
| `mask_report/masks.json` | Depth source, estimated floor plane, all parameters used, per-camera coverage statistics. |
| `mask_report/mask_sheet.jpg` | QA sheet: 5 sample frames × all cameras, background dimmed, mask outline in green. |

Measured on the `professor` Azure Kinect capture (6 cameras, 1080×1920, 377 frames):
all masks in 83 s on the CPU. The subject covers about 9–10% of each image. The
per-camera floor offsets were 0.1–2.1 cm.

Source independence was checked on the same capture: the Kinect depth was converted
to plain registered depth maps at half resolution, then run through the generic
`--depth-source registered` path. The resulting masks matched the Kinect adapter's
with IoU 0.98 on average (worst camera and frame: 0.94), with the same floor plane and
stage. `tests/test_depth_masks.py` covers the same property on synthetic rigs:
identical masks under an arbitrary rotation and translation of the world.

## How it works

Setup, from sample frames (5 spread over the range):

1. **Floor and up direction.** The depth of all cameras is fused in world space and a
   RANSAC plane is fitted. The plane normal must be within 25° of the mean camera
   image-up direction and the plane must lie at least 0.3 m below every camera. No
   world-frame convention is assumed.
2. **Stage centre and radius.** Above-floor points near the cameras' common look-at
   point are binned on the floor plane; the densest blob is the subject. Its centre is
   the stage centre, and its horizontal extent plus `--radius-pad` is the stage radius.
3. **Per-camera floor bias.** Each camera's median floor height on a ring just outside
   the stage. ToF sensors disagree by 1–2 cm, which matters against a 2.5 cm floor
   margin. Disable with `--no-floor-bias`.

Per frame:

1. Every camera's depth is unprojected and moved to world space with its pose.
2. **Stage cylinder.** A point is foreground when it lies within the stage radius of the
   centre and between `--floor-margin` and `--top` above the floor.
3. **Column support.** Foreground points lower than `--low-band` are kept only where the
   same floor cell (`--column-cell`, 4 cm) also holds foreground above the band. Feet
   and chair legs survive; cables and floor noise do not.
4. **Main 3D cluster.** The fused foreground is voxelised (`--cluster-voxel`, 5 cm).
   The largest 26-connected cluster is kept, plus any cluster at least
   `--cluster-keep-ratio` (15%) of its size. Stray clumps are dropped.
5. **Projection with occlusion test.** The fused foreground of *all* cameras is
   projected into each camera, so a hole in one sensor's depth is filled by the others.
   A point is dropped in a camera whose own depth shows background more than
   `--occlusion-margin` (15 cm) in front of it, e.g. a light stand between camera and
   subject.
6. **2D clean-up.** Splat (`--splat-px`), morphological close (`--close-px`), fill of
   enclosed holes up to `--max-hole` of the image area (depth dropouts; larger enclosed
   regions such as floor between chair legs stay background), removal of components
   smaller than `--min-component`, safety dilation (`--dilate-px`) and a Gaussian
   feather (`--feather-px`).

## Input contract

Every rig needs:

- `reference_export.json`: per camera `folder`, `camera_model_name`, `params`, `width`,
  `height`, and the world→camera pose `R`, `t`. Poses must be in metres (the same scale
  as the depth); no axis convention is assumed. Supported models: `SIMPLE_PINHOLE`,
  `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `FULL_OPENCV`. Undistort fisheye
  images first.
- `images/<cam>/<frame:06d>.jpg`: used to find the frame range and for the QA sheet
  (`--images-dir`, `--image-ext` to change).
- One depth source, selected with `--depth-source` (default `auto`: `k4a` when
  `k4a_cameras.json` exists, otherwise `registered`).

### Registered depth (any sensor)

`--depth-source registered` reads `depth_registered/<cam>/<frame:06d>.png`
(`--depth-dir` to change):

- uint16, `--depth-scale` metres per unit (default 0.001, i.e. millimetres), 0 = invalid;
- or `<frame:06d>.npy`, float32 metres (0 or non-finite = invalid);
- **z-depth along the colour camera's optical axis**, registered to that camera's colour
  image (same viewpoint and intrinsics as in `reference_export.json`), not the depth
  sensor's own view and not the ray length;
- any resolution with the image's aspect ratio; the intrinsics are scaled to it.

To support a new sensor, write its depth in this form; the rest of the script does not
change.

### Azure Kinect

`--depth-source k4a` (selected automatically) reads the raw depth written by
`prep/extract/extract_azure_kinect.py extract`: `depth/<cam>/<frame:06d>.png` (uint16
millimetres, depth-camera view) and the factory depth intrinsics and depth→colour
transform from `k4a_cameras.json`. No registration step is needed.

## Assumptions

- Cameras are roughly upright: image "up" points away from the floor. This picks the
  floor among candidate planes. Images rotated with `--rotate` during Kinect extraction
  satisfy it.
- The floor is visible in the depth. Without a floor, pass `--no-floor`: the cylinder
  then has no height limits, up is the mean camera image-up, and column support and
  floor bias are off. The stage estimate then counts every point within
  `--search-radius` of the look-at point, so you may need
  `--center` / `--radius`.
- One subject (or one connected group) near the middle of the rig. Separate people far
  apart may be dropped by the main-cluster step; lower `--cluster-keep-ratio` or set
  `--cluster-voxel 0`.

## Tuning

Tune with `--qa-only`: it processes only the 5 QA frames and finishes in seconds. Look
at `mask_report/mask_sheet.jpg`, then run without `--qa-only` for the full set. (The
Windows runner does not treat a `--qa-only` `masks.json` as a finished stage.)

| Option | Default | When to change |
|---|---|---|
| `--radius`, `--center A B` | estimated | The estimate cuts off an outstretched arm or a prop, or includes a nearby object. `--center` is in ground-plane coordinates (`ground.e1`, `ground.e2` in `masks.json`); the console prints the estimated centre. |
| `--radius-pad` | 0.25 m | Added to the estimated radius. Raise it for wide motions. |
| `--floor-margin` | 0.025 m | Floor still in the mask: raise it. Shoe soles cut off: lower it. |
| `--top` | 2.3 m | Raised arms or a tall prop cut off at the top. |
| `--low-band` | 0.12 m | Cables or floor patches near the feet remain: raise it. `0` turns column support off. |
| `--dilate-px` | 6 | Safety margin around the subject. Raise it if hair or fingers are clipped. |
| `--feather-px` | 2.0 | Width of the soft edge (Gaussian sigma, px). |
| `--cluster-voxel` | 0.05 m | Raise it if a thin, poorly sensed part (a raised hand) is detached and dropped; `0` keeps every cluster. |
| `--workers` | min(6, CPUs) | Parallel frame workers. |

## QA

- `mask_report/mask_sheet.jpg`: on every tile the green outline should follow the
  subject with a small margin. No floor patches, walls, light stands or cables inside
  it; no missing hands, feet or hair.
- `mask_report/masks.json` → `cameras.<cam>`: `fg_fraction_median/min/max` give the
  coverage per camera. `frames_with_jump_gt_25pct` lists (up to 50) frames whose
  coverage changed by more than 25% from the previous frame. Open those masks: a jump is
  either fast motion or a mask failure.
- The console prints the floor fit (camera heights should match the rig), the estimated
  stage and the per-camera floor offsets. Offsets of several centimetres point to a
  calibration or depth problem.

## How the masks are used

Polarity: 255 = subject, 0 = background. Soft values are used as they are. Masks must
be the same size as the images and in the same image space. The pipeline's images are
undistorted PINHOLE, so the masks are too.

**Initialisation** (`prep-roma-trigs-memory.py --masks-root masks`, file pattern
`--mask-pattern {frame:06d}.png`): RoMa match certainty is multiplied by the mask
(mask < 0.5 counts as 1e-4). A triangulated point is kept only if at least 75% of the
cameras that see it vote foreground (`--mask-min-fg-fraction 0.75`,
`--mask-min-fg-alpha 0.5`). Floor and wall points are therefore not initialised.

**Training** (`trigs-train-colmap`):

- `--masks-root masks` alone only adds masked PSNR metrics.
- `--masks-root masks --config.masked-rgb-loss` trains with the masks. Inside the mask
  the normal RGB loss applies. Outside, the prediction is replaced by the ground truth
  (no RGB gradient) and alpha losses push opacity to 0:
  `--config.lambda-alpha-outside-mask` (default 0.1) and a BCE term
  `--config.lambda-alpha-mask` (default 0.5). The trainer README's masked recipe for
  `trigs-train-plys` uses `--config.lambda-alpha-outside-mask 1.0
  --config.lambda-alpha-mask 0.05`; it has not been tuned for this data.
- Do not add `--config.masked-rgb-composite`: it also requires `--foregrounds-root`
  (edge-decontaminated foreground images), which this script does not produce.
- The trainer stops with an error if a frame in the training window has no mask. Mask
  every frame you train on, or pass `--no-error-on-missing-mask`.

Masked and unmasked initialisation produce different keyframes. Keep them in separate
files (the Windows runner writes `keyframes_masked_f<start>-<end>.npz` vs
`keyframes_f<start>-<end>.npz`).
