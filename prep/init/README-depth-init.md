# Point initialisation from depth

`prep/init/depth_init.py` builds the TriGS keyframe NPZ from the metric depth of every
camera. It writes the same NPZ schema as `prep-roma-trigs-memory.py`, so
`trigs-train-colmap --npz-path` reads it unchanged. It needs no GPU and no matching
network.

Compared with the RoMa initialisation it gives about 4x more points per keyframe
(60,000 by default against ~14,000). The points are metrically placed and spread
evenly over every surface that any depth camera sees. The RoMa points cluster where
image matches are good.

Depth is read through the source-independent depth sources of `prep/mask/depth_masks.py`
(`--depth-source auto|registered|k4a`; see `prep/mask/README-depth-masks.md`, "Input
contract"). The script contains nothing specific to one sensor.

## Run

From the capture data directory:

```bash
python "$REPO/prep/init/depth_init.py" --data-dir . --frame-start 1 --frame-end 31 \
    --output-path keyframes_depth_f0001-0031.npz
```

Then train with it. Pass a per-keyframe budget at least as large as `--target-points`
(see [Trainer flags](#trainer-flags)):

```bash
trigs-train-colmap --data-dir . --output-dir runs/depth --npz-path keyframes_depth_f0001-0031.npz \
    --masks-root masks --config.masked-rgb-loss --config.target-count-per-keyframe 60000
```

Inputs: `reference_export.json`, `images/<cam>/<frame:06d>.jpg`,
`masks/<cam>/<frame:06d>.png` (255 = subject), one depth source, and optionally
`sync.json`. These are the same files the mask stage and prep-roma use.

## What it does

For every keyframe t (`--frame-start`, then every `--keyframe-step` frames up to
`--frame-end`):

1. **Depth per camera.** Each camera's depth is z-buffered into its colour image at
   `--work-scale` (default 0.5, i.e. 540×960 for 1080×1920 images). Empty pixels whose
   8 neighbours are mostly valid and agree in depth are filled.
2. **Flying pixels.** A pixel is dropped when any pixel within `--edge-radius` (1 px)
   differs in depth by more than `--edge-abs + --edge-rel × z` (3 cm + 1% of depth), or
   when it lies within `--hole-erode` (1 px) of a hole. This removes the mixed-depth
   pixels that ToF sensors produce at silhouettes.
3. **Foreground.** Pixels inside the camera's own mask (≥ `--mask-min-fg-alpha`, 0.5)
   are unprojected to world space. Each point is then projected into every camera. A
   camera votes when the point falls inside its image and is not occluded: the
   camera's own depth is not more than `--occlusion-margin` (5 cm) in front of the point.
   The point is kept when at least `--min-votes` (2) cameras vote and at least
   `--mask-min-fg-fraction` (75%) of them see it inside their mask (≥ 0.5). This is
   the prep-roma vote plus an occlusion test. Finally, only the largest 3D-connected
   cluster survives, plus any cluster with at least 15% of its size. This is
   `depth_masks.main_cluster`, with `--cluster-voxel` 5 cm and `--cluster-keep-ratio` 0.15.
4. **Colour.** Per-channel median of the bilinearly sampled images of every camera
   whose depth agrees with the point within `--colour-margin` (3 cm). The source camera
   always counts. On the test capture a point is coloured from 2.65 views on
   average.
5. **Velocity.** Per camera, dense DIS optical flow (OpenCV, CPU) from t to t+d and
   from t to t−d, each checked forward-backward (`--fb-max-px`, 1 work pixel). Both
   ends are lifted through the depth. A track is rejected if the target pixel failed
   the edge filter or the depth changed by more than `--max-dz` × d. The velocity is
   (X(t+d) − X(t−d)) / 2d where both ends exist. At the first and last keyframe of the
   range only one end exists. d = `--velocity-delta`; the default is
   (keyframe step + 1) // 2 = 3. Samples faster than `--max-speed` (0.1 m/frame) are
   discarded. The acceleration (X(t+d) − 2X(t) + X(t−d)) / d² is stored too.
6. **Sync.** If `sync.json` exists (`--sync-json`; `none` disables it), each
   camera's points are moved from that camera's capture time to the nominal frame time:
   X − v × offset_frames. Times stay exactly on the keyframe grid. This matches
   prep-roma with `--sync-json` (its default motion-compensated rows) and the
   trainer, which reads the same `sync.json` for the image times.
7. **Fusion and downsampling.** A cubic voxel grid is sized by bisection so that at
   most `--target-points` voxels are occupied. Per voxel the script stores the centroid
   position, the mean colour, and the per-component median of the velocities of its
   samples. Voxels without a velocity sample take the median of up to 8 measured
   voxels within `--velocity-inherit-radius` (5 cm). Then a k-NN median
   (`--velocity-smooth-k`, 8) smooths the velocity field.

Frames outside `--frame-start..--frame-end` are never read, including for flow.

## NPZ schema

Identical keys, dtypes and shapes to `prep-roma-trigs-memory.py`. N = total rows.

| Key | dtype, shape | depth_init content |
|---|---|---|
| `positions` | float32 [N,3] | voxel centroid, world metres, at the nominal keyframe time |
| `colors` | float32 [N,3] | RGB in [0,1] |
| `velocities` | float32 [N,3] | world units per video frame (`velocity_units`) |
| `accelerations` | float32 [N,3] | per video frame²; 0 where `has_acceleration` is False |
| `has_acceleration` | bool [N] | both flow ends measured |
| `track_flags` | uint8 [N] | 1 = forward end measured, 2 = backward end (bits as `rigid_motion.TRACK_FWD/BWD`); 0 for inherited velocities |
| `times` | float32 [N,1] | (keyframe − source_frame_start) / (source_frame_end − source_frame_start) |
| `durations` | float32 [N,1] | same normalisation; prep-roma's speed-adaptive rule: step/(end−start) × clip(1/(1+speed/median speed), 0.2, 1) |
| `has_velocity` | bool [N] | measured or inherited; False rows have velocity 0 |
| `is_static` | bool [N] | all False (every row is a dynamic keyframe row) |
| `dynamic_protect` | bool [N] | all False |
| `dynamic_allowed` | bool [N] | all True |
| `stage_keep` | bool [N] | all True |
| `source_frames` | int32 [N,1] | keyframe index (source image frame) |
| `frame_start`, `frame_end` | int32 scalars | logical range 0 .. end−start |
| `source_frame_start`, `source_frame_end` | int32 scalars | `--frame-start`, `--frame-end` |
| `n_keyframes`, `keyframe_step` | int32 scalars | |
| `keyframe_frames` | int32 [K] | the keyframe indices |
| `fps` | float32 scalar | `--fps` (default 30) |
| `velocity_units` | str | `world_per_video_frame` |
| `time_units` | str | `normalized_0_1_over_source_frame_range` |
| `mode` | str | `depth_init_dynamic` (`..._merged` with `--merge-npz`) |

Not written: `local_stride` and `subkeyframe_*`, because there are no sub-keyframe
rows. Also not written: `kind`, whose presence would switch on the trainer's anchor
handling. There is no scale key: the trainer sets every Gaussian's initial scale from
k-NN spacing (`--config.init-scale-knn-factor` × mean distance to 4 neighbours). The
trainer uses `source_frames` (not `times`) for timing when `--start-frame` /
`--end-frame` are set, which `resolve_npz_defaults` always does.

## Trainer flags

- `--config.target-count-per-keyframe 60000` (or ≥ `--target-points`). The default
  is 25,862: the trainer randomly subsamples every keyframe to that many dynamic rows.
  With the default you lose 57% of the depth points.
- `--start-frame` / `--end-frame` / `--keyframe-step` / fps default to the NPZ's
  values. An explicit `--keyframe-step` must equal the NPZ's.
- Masks: `--masks-root masks --config.masked-rgb-loss`, as for the masked RoMa NPZ.
- `--config.init-sigma-t-mode` (default `npz`) uses the NPZ durations clamped to
  [0.4, 1] × keyframe_step / fps.
- GPU memory at initialisation: `from_colmap._radius_ball_centroid` builds a
  2048 × N distance matrix per keyframe. At N = 60,000 that is a transient ~1.5 GB
  (470 MB per array); it scales linearly with `--target-points`. Sharing the 8 GB GPU
  with another training run that held 5.6 GB, it ran out of memory there. On its own
  GPU this is not a concern.

### Combining with RoMa

- `--merge-npz keyframes_masked_f0001-0090.npz` appends another prep NPZ's rows to the
  output. It takes the rows at this run's keyframes plus all static rows. They are
  re-timed and their durations rescaled to this run's frame range. Velocity units and
  fps must match. Off-grid sub-keyframe rows are dropped.
- The trainer's `--hybrid-anchor-npz A --hybrid-coverage-npz B` is not a union. Per
  keyframe it takes the anchor NPZ's dynamic rows when there are any, and falls back to
  the coverage NPZ only otherwise. With the depth NPZ as anchor, RoMa would only fill
  keyframes the depth NPZ lacks, which never happens for the same range. Use
  `--merge-npz` for a union.

## Options

| Option | Default | Notes |
|---|---|---|
| `--target-points` | 60000 | Max points per keyframe. The voxel size follows from it (7.2 mm on the test capture). |
| `--keyframe-step` | 5 | Same as prep-roma. |
| `--depth-source`, `--depth-dir`, `--depth-scale` | auto | As in `depth_masks.py`. |
| `--depth-offsets-json` | — | Per-camera additive depth corrections from `prep/extract/k4a_depth_offsets.py` (see [Depth offsets](#depth-offsets)). Off by default. |
| `--masks-root`, `--mask-pattern` | masks | `none` disables masks; the result then contains the whole room. |
| `--work-scale` | 0.5 | Depth, edge filter and flow grid relative to the colour image. |
| `--velocity-delta` | 0 = (step+1)//2 | Flow baseline in frames. Depth noise in the velocity falls as 1/d (measured below). A larger d also smooths over fast motion. |
| `--velocity-mode` | centred | `forward` uses only t → t+d where it exists. |
| `--flow-preset` | medium | DIS preset: ultrafast, fast, medium. |
| `--velocity-smooth-k` | 8 | 1 turns smoothing off. |
| `--min-votes` | 2 | 1 keeps points that only their own camera can see. On the test capture these were background pixels at the dilated mask border, up to 3.6 m from the subject. |
| `--cluster-voxel` | 0.05 | 0 keeps every 3D cluster. |
| `--export-ply-dir` | — | Writes `depth_init_f<frame>.ply` per keyframe (xyz, rgb, vx vy vz). |
| `--stats-path` | — | JSON with per-keyframe counts, speeds and timings. |
| `--workers` | min(6, CPUs) | Threads over cameras. |

## Depth offsets

`--depth-offsets-json depth_offsets.json` reads `offsets_m` (metres, keyed by camera
folder) from the output of `prep/extract/k4a_depth_offsets.py` (see
[README-azure-kinect.md](../extract/README-azure-kinect.md#per-camera-depth-offsets)). It
adds each camera's offset to the colour-camera z of every valid depth point at load:
the point moves along its colour ray, and invalid depth stays invalid. This works for
every `--depth-source`. A camera missing from the file is an error. The NPZ records the
offsets as `depth_offsets_m` (float32 [C]) and `depth_offsets_cameras` (folder names);
`--stats-path` records them too. Without the option nothing changes.

On the test capture (frames 1–31, `masks_refined`, 60,000 target points), the
corrected run (`keyframes_depth_refined_doff_f0001-0031.npz`) fuses into a voxel of
6.8–6.9 mm against 7.0–7.1 mm without offsets: the six cameras' surfaces coincide
better. Both runs took 14 s.

## Measured on the test capture

6 Azure Kinects, 1080×1920, frames 1–31 (7 keyframes), CPU only (shared with a
running training job):

- **Time:** 15–22 s in total, 1.9–3.8 s per keyframe (load 0.4–1.0 s, flow 0.2–0.6 s,
  vote and colour 0.6–1.3 s, fusion 0.6–1.1 s). Peak memory 0.66 GB.
  `--depth-source registered` (540×960 PNG): 26.5 s.
- **Points:** about 211k candidates → 188k after the vote → 59.8k–60.0k points per
  keyframe (voxel 7.1–7.2 mm); 419,353 rows in total. The bounding box is
  x −0.45…0.44, y −0.03…1.36, z −0.52…0.39 m (RoMa: x −0.43…0.42, y −0.02…1.35,
  z −0.51…0.39). The farthest point is 0.56 m from the subject's vertical axis. Without
  `--cluster-voxel`, a small white object on the floor 1.5 m away survived (157
  points); it lies inside the masks.
  99.2–99.9% of the points project inside each camera's mask.
  The points hit 80–85% of the mask area (4×4-pixel cells; the masks are dilated by
  6 px plus a feather).
- **Against RoMa** (`keyframes_masked_f0001-0090.npz`, same keyframes): median
  distance from a RoMa point to the nearest depth point is 5.7 mm (p90 18 mm). On
  RoMa's fastest 25% of points, the median cosine between the two velocities is
  0.84–0.94 (0.40 at frame 1, where RoMa's p90 speed is only 0.3 mm/frame). The median
  colour difference is 0.024.
- **Velocity:** 98–99% of points measured, the rest inherited (about 8% at the last
  keyframe, which has only a backward end); 0–41 zero per keyframe. Speed p50
  0.6–0.9 mm/frame, p90 1.8–4.0, p99 3.5–10.
- **Noise floor** (masks off, static walls and floor 1–2.5 m from the subject, no
  smoothing): median speed 2.2 mm/frame with d = 1 and 0.75 mm/frame with d = 3
  (centred). One-sided ends (first and last keyframe) are about 2× noisier. Hence
  the default d = 3. On a mostly still seated subject, most of the remaining ~0.6 mm/frame
  median is still noise. It amounts to about 1.5 mm of drift over a keyframe's ±2.5 frames.
- **Source independence:** `--depth-source k4a` and `--depth-source registered`
  give point sets within 3.4 mm (median nearest-neighbour distance, p90 5.4 mm) and
  velocities within 0.13–0.26 mm/frame (median difference).
- **Trainer loader:** `resolve_npz_defaults` reads frames 1–31, step 5, fps 30 from the
  file. `build_canonical_from_npz` keeps all 419,353 rows at 7 keyframe times. Initial
  scale p50 is 1.95 mm (p10–p90 1.7–2.3 mm), and sigma_t p50 is 0.083 s.
- **Smoke training** (alone on the RTX 8 GB GPU, frames 1–31, `--masks-root masks
  --config.masked-rgb-loss`, 1000 steps, PSNR on the trainer's 42 fixed eval pairs):

  | Init | Primitives | Eval masked PSNR, step 0 → 500 | Wall time |
  |---|---|---|---|
  | depth_init (`--config.target-count-per-keyframe 60000`) | 419,353 | 12.0 → 20.1 dB | 8.8 min |
  | RoMa `keyframes_masked_f0001-0090.npz` | 95,424 | 11.3 → 20.3 dB | 6.0 min |

  The trainer did not log an eval at step 1000. These are training views, and 500
  steps say nothing about the novel-view blur this initialisation is meant to fix.
  That needs a full-length run with a held-out camera.

## Tests

`prep/init/tests/test_depth_init.py` builds a synthetic rig: 6 ray-cast cameras, a
textured sphere translating 13 mm/frame, a floor, mixed-depth flying pixels on one
camera, and a small detached object inside the masks. It checks the following:

- the schema, dtypes and timing;
- points on the sphere: p99 surface error < 8 mm, max < 2 cm, no floor and no flying pixels;
- colours against the analytic texture;
- velocity within 1.5 mm/frame of the truth, with d = 1 and d = 2;
- the detached object removed;
- sync compensation (cameras captured half a frame late);
- `--merge-npz`;
- invariance under an arbitrary rigid transform of the world;
- that the trainer's `build_canonical_from_npz` accepts the file.

```bash
python -m pytest prep/init/tests/test_depth_init.py -q
```

## Notes

- The worker threads call numpy concurrently. A multi-threaded OpenBLAS hung or
  crashed intermittently under that on Windows. The script therefore sets
  `OPENBLAS_NUM_THREADS=1` (also OMP and MKL) before importing numpy. If numpy was
  already imported (the script imported as a module), it runs with `--workers 1`.
- Everything outside the masks is excluded, including static background. The NPZ has no
  static rows. The masked training loss pushes opacity outside the mask to zero anyway.
