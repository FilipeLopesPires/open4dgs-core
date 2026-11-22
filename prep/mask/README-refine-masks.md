# Refining coarse masks with colour

`prep/mask/refine_masks.py` shrinks loose foreground masks to the subject's colour edge.
The input is any coarse mask that contains the whole subject: typically the depth masks
from `depth_masks.py`, but any source with the same layout works. Registered depth and a
SAM 2 person segmentation are optional extra cues.

Why: depth masks are deliberately generous (splat, close and a 6 px safety dilation, plus
calibration error between sensors). On the `professor` capture they include floor around
the feet and chair legs, pieces of wall and light stands next to the head, and a red object
above the head in camera 0001. The trainer learns those patches as opaque splats (halo and
floaters). The refined masks follow the silhouette of person, chair and laptop with a 2–3 px
soft edge.

## Run

From the capture data directory, after `depth_masks.py`:

```bash
python "$REPO/prep/mask/refine_masks.py" --data-dir .
```

Every frame that has both an image and a coarse mask in all cameras is processed. Use
`--frame-start` / `--frame-end` (inclusive) or `--frames 1-31,100,200` for a subset, and
`--qa-only` to process only the QA frames (5 frames spread over the range, or
`--qa-frames`).

Outputs (the coarse masks are never modified; `--output-dir masks` is refused):

| Path | Contents |
|---|---|
| `masks_refined/<cam>/<frame:06d>.png` | 8-bit alpha mask, same size and polarity as the coarse mask: 255 = subject, 0 = background, soft values on a 2–3 px edge. |
| `masks_sam/<cam>/<frame:06d>.png` | SAM 2 person masks (0/255), when SAM 2 is used. A cache: existing files are reused on the next run. Delete the folder after changing `--sam-model`. |
| `mask_report/refine.json` | Parameters, floor source, SAM settings, timing, per-camera statistics. |
| `mask_report/refine_sheet.jpg` | QA sheet: one row per QA frame, one tile per camera, cropped to the coarse mask. Background dimmed (blue) by the refined mask, coarse outline red, refined outline green. |

Measured on the `professor` capture (6 cameras, 1080×1920, 377 frames, 2262 masks):
see [Results](#results-on-the-professor-capture).

## Inputs

| Input | Required | Notes |
|---|---|---|
| `images/<cam>/<frame:06d>.jpg` | yes | `--images-dir`, `--image-ext`. |
| `masks/<cam>/<frame:06d>.png` | yes | Coarse mask, 8-bit, > 127 = subject. `--masks-dir`. Must contain the whole subject; the refined mask can extend it only by `--max-grow-px` (and, where the person mask says so, by `--person-grow-px`). |
| `depth_registered/<cam>/<frame:06d>.png` or `.npy` | no | Same contract as `depth_masks.py --depth-source registered`: z-depth along the colour camera's optical axis, registered to the colour image, uint16 × `--depth-scale` metres (default millimetres) or float32 metres, 0 = invalid, any resolution with the image's aspect ratio. `--depth-dir`; `--no-depth` ignores it. |
| `reference_export.json` | with depth | Intrinsics and world→camera poses in metres (same file as for `depth_masks.py`). |
| `mask_report/masks.json` | no | Written by `depth_masks.py`: floor plane, per-camera floor offsets and stage cylinder. Without it, the floor is fitted from the registered depth (RANSAC, as in `depth_masks.py`); if that fails, only the "behind the subject" depth test is used. `--ground-json`. |
| SAM 2 | no | The `sam2` package and a checkpoint (see [SAM 2](#sam-2)). Without it the method is colour + depth only. |

Nothing in the script is specific to a sensor: an RGB-only rig can refine any coarse mask
(for example a dilated SAM 2 or Background Matting mask) with `--no-depth`.

## How it works

Per camera and frame, at half resolution (`--work-scale 0.5`):

1. **SAM 2 person mask** (when available): the SAM 2 image predictor, prompted with the
   bounding box of the coarse mask (10 px margin). It segments the *person*; the laptop and
   the chair are usually not part of it. It runs once per frame in a separate process on the
   GPU (if at least 2.5 GB are free, else on the CPU) and is cached in `masks_sam/`.
2. **Depth cues** (when registered depth is available). A pixel with valid depth is
   - *background*: outside the stage cylinder (+ `--stage-margin`) or more than
     `--behind-margin` (0.25 m) behind the farthest subject surface within
     `--local-radius-px` (subject depth from the eroded coarse mask, ignoring floor pixels
     and anything more than 1 m behind the median subject depth). Removes walls and stands.
   - *floor candidate*: less than `--floor-margin` (3 cm) above the floor. Not background
     by itself: Azure Kinect depth puts shoe toes at 0–3 cm and the floor scatters by
     ±3 cm, so depth cannot tell a shoe from the floor it stands on.
   - *subject*: inside the coarse mask, more than 5 cm above the floor, not background.
3. **Trimap for GrabCut.**
   - Sure foreground: the coarse mask eroded by `--core-erode-px` (12 work px = 24 image
     px) minus depth background and floor candidates; the eroded depth-subject pixels; the
     person mask eroded by `--person-erode-px`.
   - Sure background: outside the coarse mask (dilated by `--max-grow-px`); eroded depth
     background; and, when both depth and SAM 2 are used, *unclaimed* pixels: more than
     `--unclaimed-gap-px` (4 work px) outside the person mask, with no subject depth
     nearby and outside the eroded core. These are typically dark wall next to dark hair
     where the ToF depth is invalid, which colour alone cannot remove.
   - Floor candidates: initial label from a colour model of the floor just outside the mask
     versus the subject below 0.5 m (shoes, trousers); clear floor colour
     (P(subject) < `--floor-sure-bg-prob`) is sure background.
   - The person mask may extend the coarse mask by up to `--person-grow-px` (20 work px)
     where depth does not say background: depth masks often clip shoe toes at the floor
     margin.
   - Everything else is "probably foreground" (or "probably background" if outside the
     person mask and without subject depth).
4. **GrabCut** (OpenCV, `--grabcut-iters` 4) labels the uncertain pixels from colour models
   and contrast-sensitive smoothness, in a box around the coarse mask with `--roi-pad-px`
   of background context.
5. **Temporal vote**: the binary results of `--temporal` frames on each side (default 1)
   are averaged; > 0.5 wins (majority of 3 frames). A label that flips for one frame is
   overruled. First and last frames use their own result.
6. **Full resolution**: the vote is upsampled, snapped to image edges with a guided filter
   (`--guided-radius` 4 px, `--guided-eps`), thresholded at 0.5 and cut to the allowed area.
   Only components that contain sure foreground are kept, enclosed holes up to 0.3% of the
   image are filled, then a `--dilate-px` (1 px) margin and a Gaussian feather of sigma
   `--feather-px` (1.0 px) are applied. Pixels inside the hard mask stay ≥ 128.

## How the runner and trainer use it

The refined masks have the same layout, size and polarity as `masks/`, so every consumer
takes them through its existing mask-root option:

- **Initialisation**: `prep-roma-trigs-memory.py --masks-root masks_refined` (and
  `--mask-pattern {frame:06d}.png`, the default layout); `depth_init.py --masks-root
  masks_refined`.
- **Training**: `trigs-train-colmap --masks-root masks_refined --config.masked-rgb-loss`
  (see the training section of [README-depth-masks.md](README-depth-masks.md#how-the-masks-are-used)
  for the loss options). The trainer stops if a training frame has no mask, so refine
  every frame you train on.

Keyframes initialised with different masks differ: keep them in separate files (for
example `keyframes_refined_f0001-0031.npz`).

The coarse masks are still the right input for `refine_masks.py` itself and remain the
safer choice when a subject part is thin, fast and poorly coloured (see
[Limitations](#limitations)).

## Multi-view consistency pass (optional)

`prep/mask/consistent_masks.py` is an opt-in post-pass over the refined masks that uses all
cameras of a frame at once (`refine_masks.py` itself is unchanged). Each camera decides on
its own, so a part that one camera drops is learned as transparent from that view: on
`professor` (frames 1, 40, 90) 5.3% of the masked depth points of one camera land outside
another camera's refined mask, 11% at the top of the head (hair against a dark wall, no
ToF depth on dark hair) and 13% at the chair seat height (black chair shell: no depth at
all, so the coarse depth mask has a hole that refining cannot fill; laptop lid edges).

```bash
python "$REPO/prep/mask/consistent_masks.py" --data-dir . --frames 1-90 --unknown --workers 3
python "$REPO/prep/mask/straight_foregrounds.py" --data-dir . --masks-dir masks_consistent \
    --output-dir foregrounds_consistent --workers 4
```

Per frame, at depth resolution:

1. **Source points**: each camera's registered depth inside its mask (eroded by
   `--src-erode-px`), edge-safe (3×3 depth range < `--edge-tol` 3 cm, no flying pixels)
   and in the subject volume (above the floor by `--floor-margin`, inside the stage +
   `--stage-margin`), back-projected to world.
2. **Support**: the points of the other cameras are projected into camera j and splatted
   (`--splat-px` closing). A camera *supports* a pixel when its points agree with j's depth
   (`--depth-tol` 4 cm) or j has no depth there; points in front of j's measured surface
   (j sees the wall through them) are *weak*; points behind it are occluded.
3. **Restore** (outside j's mask, within `--band-px` 16 px of it, components touching the
   mask): ≥ `--min-views` (2) supporting cameras, or j's own edge-safe subject-volume depth
   (refine's depth-subject test: above floor, inside the stage, not behind the local
   subject) confirmed by ≥ 1 camera (`--keep-volume`, default on). Snapped to colour edges
   with the same guided filter and feather as `refine_masks.py` and merged; the input mask
   is never reduced. `--min-width-px` > 0 drops thin slivers.
4. **Unknown** (`--unknown`): pixels in the band with 1 supporting camera or weak support
   only, and enclosed no-depth blobs next to the mask (no image border, < 2% of the image
   (`--nodepth-max-frac`), ≥ 15% of the rim on the mask, almost no background depth above
   floor level on the rim: the chair shell). Each kind can be switched off
   (`--no-unknown-weak`, `--no-unknown-nodepth-holes`).

Outputs: `masks_consistent/<cam>/<frame:06d>.png`, `mask_report/consistent.json`
(per-camera area gain, unknown fraction, temporal IoU; `params.dilate_px` lets
`straight_foregrounds.py` find the core erosion) and `consistent_sheet.jpg` (added pixels
green, unknown magenta, input outline white). About 0.75 s per frame with 3 workers on the
CPU (~5 min for 377 frames); each worker holds one frame of all cameras (images, masks,
depth and full-resolution float maps: a few hundred MB).

### Mask file contract: grey + confidence alpha

Without `--unknown` every mask is a plain 8-bit greyscale PNG. With it, a frame with unknown
pixels is an 8-bit grey+alpha PNG (PIL mode `LA`): grey = the mask as before (0 background,
255 subject, soft edge), alpha = confidence (255 known, 0 unknown, intermediate = soft
weight); unknown pixels always have grey < 128. A plain greyscale PNG means "all known".
Readers that load with `cv2.IMREAD_GRAYSCALE`, `IMREAD_COLOR` or PIL `.convert("L")` see the
plain mask (all of `prep/mask`, `prep/init`, `prep/align`, `prep/extract`).
`downscale_mask_outputs.py` keeps the alpha (both channels shrunk with INTER_AREA). Code
that reads with `cv2.IMREAD_UNCHANGED` gets 4 channels (B = G = R = grey, A = confidence),
and raw PIL / imageio arrays have 2 channels; such readers must take channel 0 as the mask.
The trainer ignores unknown pixels in its mask and composite losses.

### Results (professor, frames 1–90)

Silhouette consistency (fraction of camera-i masked depth points, not occluded, outside
≥ 1 other mask; frames 1, 40, 90):

| | Overall | 0001 | 0002 | 0003 | 0004 | 0005 | 0006 | Head band | Chair band |
|---|---|---|---|---|---|---|---|---|---|
| `masks_refined` | 5.3% | 2.6 | 4.0 | 6.2 | 5.2 | 7.6 | 6.5 | 11% | 13% |
| `masks_consistent` | 4.2% | 1.7 | 3.5 | 5.7 | 4.3 | 5.4 | 4.6 | 8% | 12% |
| … unknown pixels not counted | 1.3% | 1.0 | 0.8 | 1.1 | 1.5 | 2.0 | 1.5 | 2% | 1% |

Frames 1, 10, 20, 31, 60: 5.3% → 4.1% (1.2% with unknown not counted). Area gain per camera
(median over frames 0.0–0.4%, max 0.9%); unknown 0.2–0.5% of the image (3–5% of the mask
area) for cameras 0001, 0002, 0005, 0006 and 1.0% (15%) for 0003 and 0004, where the chair
shell has no depth. Temporal IoU unchanged (within 0.001). `--min-views 1` restores more
(+2% area in 0003/0004, 5.3% → 2.7%, mostly real chair and hair edges on the QA sheet), but
a single view cannot tell a miss from calibration spill, so the default labels it unknown.

## SAM 2

Install once into the pipeline environment (CUDA extension not needed):

```bash
git clone https://github.com/facebookresearch/sam2.git /path/to/sam2
cd /path/to/sam2
SAM2_BUILD_CUDA=0 uv pip install --python <venv python> --no-build-isolation -e .
curl -fL -o checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

The checkpoint is found in `$SAM2_CHECKPOINT_DIR`, the `checkpoints/` folder of the
editable checkout, `~/sam2_repo/checkpoints` or `~/checkpoints`, or passed with
`--sam-checkpoint`. `--sam auto` (default) uses SAM 2 when both are found, `--sam on`
requires it, `--sam off` disables it. `--sam-model` selects `large` (default), `base_plus`,
`small` or `tiny`. The large model took 0.3 s per image on an 8 GB RTX shared with a
training job, about 4.5 s per image on the CPU; `small` was about 1.2 s on the CPU and
segmented fragments of the laptop and chair into the person, which the large model did
not.

## Tuning

Use `--qa-only` (or `--frames` with a few frames) and look at
`mask_report/refine_sheet.jpg`.

| Option | Default | When to change |
|---|---|---|
| `--core-erode-px` | 12 (work px) | Must exceed the looseness of the coarse mask (here up to ~20 work px, but floor and far pixels are removed from the core by depth anyway). Raise it if background patches stay in; lower it for a tight coarse mask. |
| `--max-grow-px` | 2 (work px) | How far the refined mask may extend beyond the coarse mask. |
| `--person-grow-px` | 20 (work px) | Extension allowed where the SAM 2 person mask extends the coarse mask (toes clipped by the depth floor margin). 0 disables it. |
| `--unclaimed-gap-px` | 4 (work px) | 0 disables the unclaimed-pixel rule. Raise it if hair is clipped where SAM 2 cuts it tight; lower it if slivers of dark wall stay next to the hair. |
| `--behind-margin` | 0.25 m | Lower it to remove background close behind the subject; raise it if a body part far from the rest (outstretched arm) is cut. |
| `--floor-margin` | 0.03 m | Floor-candidate height. |
| `--temporal` | 1 | Frames on each side of the vote; 0 disables it. |
| `--dilate-px`, `--feather-px` | 1, 1.0 | Safety margin and soft-edge width at full resolution. |
| `--workers` | min(8, CPUs) | Parallel CPU workers (each processes `--chunk` consecutive frames of one camera). Each needs about 200 MB. |

## QA

- `refine_sheet.jpg`: the green outline should follow person, chair and laptop with no
  floor, wall or stand inside it, and keep fingers, hair and shoe tips.
- `refine.json` → `cameras.<cam>`: `area_ratio_*` is refined / coarse foreground area,
  `temporal_iou_coarse_mean` and `temporal_iou_refined_mean` the mean IoU between
  consecutive frames for both mask sets, `frames_with_area_jump_gt_10pct` lists (up to 50)
  frames whose refined area changed by more than 10% from the previous frame.

## Results on the professor capture

All 377 frames × 6 cameras (`mask_report/refine.json`). Foreground = alpha > 127, as a
percentage of the image; temporal IoU between consecutive frames:

| Camera | Coarse fg | Refined fg | Refined / coarse area (min–max) | Temporal IoU coarse | Temporal IoU refined (min) |
|---|---|---|---|---|---|
| 0001 | 10.2% | 8.2% | 80% (77–85) | 0.957 | 0.994 (0.983) |
| 0002 | 9.4% | 8.0% | 85% (82–89) | 0.969 | 0.994 (0.972) |
| 0003 | 8.7% | 7.2% | 82% (79–84) | 0.973 | 0.991 (0.973) |
| 0004 | 9.1% | 7.5% | 83% (77–86) | 0.964 | 0.993 (0.975) |
| 0005 | 9.9% | 8.4% | 85% (83–90) | 0.968 | 0.993 (0.951) |
| 0006 | 9.8% | 7.9% | 81% (78–86) | 0.951 | 0.994 (0.983) |

No frame changed its refined area by more than 10% from the previous frame. The lowest
temporal IoUs are in frames 38–76, while the subject opens the laptop. Without the
temporal vote (`--temporal 0`, frames 1–31) the refined masks are still steadier than the
coarse ones: mean IoU 0.978–0.994 per camera against 0.949–0.977.

Choice of cues, measured on frames 1–31, 99–101, 199–201 and 299–301. "Feet" is the
share of the bottom 15% of the SAM 2 person mask (the shoes) inside the mask:

| Variant | Area vs coarse | Feet kept |
|---|---|---|
| Coarse depth mask | 100% | 80–98% per camera (the floor margin clips shoe toes) |
| Colour + depth, floor pixels as sure background (first prototype, frames 14–16 and 200 only) | 77–87% | shoes largely removed (visual): ToF depth puts them at floor height |
| Colour + depth, floor pixels as colour-decided candidates (`--sam off`) | 81–89% | 81–98% |
| Colour + depth + SAM 2 person hint (default) | 80–89% | 98–99% |

SAM 2 alone (box prompt) segments the person well, including shoes, fingers and hair, but
leaves out the laptop and the chair; extra point prompts on them gave blotchy partial
masks. It is therefore used only as a sure-foreground hint inside GrabCut.

Timing (24-core CPU, 8 GB RTX shared with a training job): SAM 2 large about 0.3 s per
image (about 11 min for 2262 images), the CPU stage 398 s with `--workers 4`
(0.43 s of CPU per mask). With the SAM 2 cache in place a re-run takes only the CPU stage.

QA: floor patches around feet and chair legs, wall and light-stand pieces next to the
head and the red object above the head in camera 0001 are gone. Fingers on the laptop,
shoe tips, the open laptop lid (also seen edge-on) and the chair seat and back are kept.

## Limitations

- The refined mask cannot recover a part that the coarse mask misses by more than
  `--max-grow-px`, except where the SAM 2 person mask covers it (up to
  `--person-grow-px`). Thin chair legs that the depth mask dropped stay dropped, and so
  does the black chair shell, which has no ToF depth in any camera (the consistency pass
  labels it unknown).
- Refining drops few pixels with subject-volume depth (30–80 edge-safe pixels per camera
  and frame at half resolution on `professor`, all decided by GrabCut colour, none by the
  unclaimed-pixel rule); `consistent_masks.py --keep-volume` restores those that another
  camera confirms.
- Without depth, background with the subject's colour next to the subject (dark wall next
  to dark hair, grey floor next to grey trousers) can stay in the mask. Without SAM 2, the
  unclaimed-pixel rule is off and shoe toes clipped by the coarse mask are not restored.
- SAM 2 is prompted with a single box, so it segments the dominant object in it. For
  several people far apart, refine with `--sam off` or provide coarse masks per person.
- Pixels within `--unclaimed-gap-px` of the person mask are decided by colour alone;
  slivers of dark background up to about 8 image px wide can remain next to dark hair.
