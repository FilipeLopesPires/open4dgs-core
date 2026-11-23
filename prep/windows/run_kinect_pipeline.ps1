<#
.SYNOPSIS
  Azure Kinect capture -> 4DGS, end to end on Windows (after setup_env.ps1).

.DESCRIPTION
  Stages (each skipped when its output exists, unless -Force; logs go to <Out>\logs\):
    inspect    audit, frame sync, QA sheets               k4a_report\inspect.json
    extract    undistorted frames + depth + sync.json      images\, depth\, sync.json
    calibrate  extrinsics from colour + depth              sparse\0, reference_export.json
    mask       foreground masks from multi-view depth      masks\, mask_report\masks.json
    init       point initialisation (RoMa v2 + MEMFOF,     keyframes[_depth][_masked|_refined]_f<start>-<end>.npz
               or measured depth with -Init depth)
    train      TriGS training (time-chunked by default)    runs\<Name>\trigs_ckpt.pt
    export     per-frame PLYs + .trigsbin timeline         runs\<Name>\plys\, runs\<Name>\<Name>.trigsbin
    asog       Animated SOG for the three.js player        runs\<Name>\<Name>.asog
               (animated-sog: installed package or -AsogRepo checkout; -AsogArgs for encoder options)

  -Preset best (default) applies the tested recipe to every option not given on the command line:
    -MasksDir masks_refined -Init depth -DepthLoss 1.0 -Composite -AlphaOutsideWeight 1.0
    -MaxAnisotropy 10 -DepthOffsets -DepthEdgePx 2, and -Steps = 165 x frames (rounded to 1000).
  Override one option by giving it (e.g. -MaxAnisotropy 4, -Composite:$false). -Preset basic runs
  the trainer defaults (depth masks, RoMa init, 30000 steps), as before.

  Added automatically when an option needs them and their output is missing:
    refine       -MasksDir masks_refined   masks shrunk to the colour edge   masks_refined\, mask_report\refine.json
    consistent   -MasksDir masks_consistent  cross-view vote + unknown band  masks_consistent\, mask_report\consistent.json
    offsets      -DepthOffsets             per-camera depth corrections      depth_offsets.json
    register     -DepthLoss                colour-registered depth maps      depth_registered\_complete.json
    foregrounds  -Composite                straight foreground images        foregrounds[_refined]\_complete.json
  foregrounds is also rebuilt when older than the mask report or not covering the frame range.

  Masks are on by default (prep\mask\depth_masks.py, see prep\mask\README-depth-masks.md):
  init drops background points (--masks-root masks) and train uses the masked RGB loss
  (--masks-root masks --config.masked-rgb-loss), which pushes opacity outside the mask to 0.
  -NoMasks skips the mask stage and runs init/train without masks (the previous behaviour).
  Masked and unmasked runs use different keyframe files, but the same run folder
  runs\<Name>: give each its own -Name to keep both.

  Training options (passed only when given; otherwise the trainer defaults apply):
    -AlphaOutsideWeight <w>  --config.lambda-alpha-outside-mask (trainer default 0.1)
    -AlphaMaskWeight <w>     --config.lambda-alpha-mask (trainer default 0.5)
    -MasksDir masks_refined  use the colour-refined masks (prep\mask\refine_masks.py, needs SAM 2)
    -Composite               --config.masked-rgb-composite --foregrounds-root foregrounds[_refined]:
                             supervise the whole frame against the subject matted on a white plate
    -MaxAnisotropy <r>       --config.max-scale-anisotropy (largest/smallest axis; 0 = off)
    -Init depth              initialise from the registered Kinect depth (prep\init\depth_init.py)
                             instead of RoMa v2 + MEMFOF; -DepthInitPoints points per keyframe
                             (default 60000), also passed as --config.target-count-per-keyframe
    -DepthLoss <w>           --depth-root depth_registered --config.lambda-depth <w> (1.0 tested)
    -TrainerSrc <dir>        run train/export with PYTHONPATH=<dir> (another trainer checkout's src)
    -TrainExtra "<args>"     appended verbatim to the trigs-train-colmap command line
    -ChunkSeconds <s>        temporal chunk window (default 3; 0 = train the whole clip at once)
    -MaxSigmaT <s>           splat lifetime cap (--config.max-sigma-t, default 0.5), applied only
                             when the clip is longer than one chunk window; 0 = off
    -AsogDevice cuda|cpu     device for the asog SH palette k-means (default cuda)
    -MasksDir masks_consistent  cross-view vote over masks_refined (prep\mask\consistent_masks.py
                             --unknown, for the frame range); disagreements are "unknown" pixels
                             the trainer leaves out of its mask losses; foregrounds_consistent
    -HoldoutViews <i,...>    --config.holdout-views: cameras (0-based, sorted folders) left out of
                             training and scored as psnr/holdout_* (novel views)
    -LearnAppearance         --config.learn-camera-appearance (per-camera colour gain + bias)
    -RandomBackground        --config.masked-rgb-random-bg (random composite plate; needs -Composite)
    -LearnTimeOffsets        --config.learn-camera-time-offsets (starts at step 8000 by default)
    -ShDegree <0-3>          --sh-degree (default 3)
    -DepthOffsets            per-camera depth corrections (offsets stage, prep\extract\k4a_depth_offsets.py
                             -> depth_offsets.json) for -Init depth (keyframes_..._doff) and -DepthLoss
    -DepthEdgePx <px>        --config.depth-edge-px (default 1; 2 recommended)
    -DepthLossFinal <s>      decay lambda_depth to lambda x s, log-linear from -DepthDecayStart
                             (5000) over -DepthDecaySteps (2000) steps; default constant
    -DepthDistortion <w>     --config.lambda-depth-distortion (from -DepthDistortionStart, default 0)
    -OpacityBinary <w>       --config.lambda-opacity-binary (from -OpacityBinaryStart, default 0)
  -DepthLoss needs a trainer with depth supervision (open4dgs-trigs-fork; see prep\windows\README.md).
  The options from -MasksDir masks_consistent on need the current open4dgs-trigs-fork trainer.

  Extract a section only (a short clip of a long capture; frames are renumbered from 1):
    -ExtractStart <n>        --start-frame: frames to skip into the synced window (default 0)
    -ExtractFrames <n>       --num-frames: frames to extract (default: the rest of the window)
  -CalibrationFrom <o4d dir> copy that take's calibration instead of solving one (same rig, cameras
                             not moved; serials must match). For short clips, where too few distinct
                             sample instants can make the solve fail (camera ring off, floor not found).

  Logs: logs\<stage>.log, except init-<keyframe file>.log, train-<Name>.log, export-<Name>.log.

  Export options (see prep\windows\README.md, "Export and file size"):
    -FromRun <run>           export runs\<run>\trigs_ckpt.pt into runs\<Name> (no training;
                             e.g. -Stages export -FromRun all3s -Name all3s_small)
    -PruneOpacity <a>        drop rows with base opacity below <a> from the export copy
                             (default 0.005; 0 keeps every row). trigs_ckpt.pt is never changed.
    -TensorDtype <policy>    .trigsbin tensor dtypes: mixed (default; fp16 except times and
                             durations), motion32, float16, float32
    -PackChunkSeconds <s>    longest .trigsbin clip (default 10): a sequence up to <s> is one
                             clip, a longer one is split into equal clips no longer than <s>
    -NoPly                   skip the per-frame PLYs (about 47 MB per frame for 1 M splats)
    -CompressPly             write the PLYs as frame_<n>.compressed.ply (PlayCanvas compressed
                             format, about 3.9x smaller; SuperSplat reads it, plain 3DGS PLY
                             readers do not)
  The export copy carries the run's training config (from the final viewer snapshot), so the
  PLYs and the .trigsbin use the temporal threshold the run was trained with.

  Review k4a_report\*.jpg and the console QA lines after inspect/calibrate, and
  mask_report\mask_sheet.jpg after mask, before spending hours on init/train
  (see prep\extract\README-azure-kinect.md).

.EXAMPLE
  # whole capture with the default recipe, up to the .asog for the three.js player
  powershell -ExecutionPolicy Bypass -File prep\windows\run_kinect_pipeline.ps1 -Capture datasets\<capture> -Rotate l
.EXAMPLE
  # stop after the masks to review mask_report\mask_sheet.jpg
  ... -Capture datasets\<capture> -Rotate l -Stages inspect,extract,calibrate,mask
.EXAMPLE
  # unmasked run next to the masked default run "full"
  ... -Capture datasets\<capture> -Rotate l -NoMasks -Name nomask
.EXAMPLE
  # only re-train and export, 45k steps, stronger opacity penalty outside the mask, new run name
  ... -Capture datasets\<capture> -Rotate l -Stages train,export -Name long -Steps 45000 -AlphaOutsideWeight 1.0
.EXAMPLE
  # best tested quality recipe (3 s): refined masks, depth init, depth loss, composite, anisotropy cap
  ... -Capture datasets\<capture> -Rotate l -Stages init,train,export -Name all3s -FrameEnd 90 `
      -Steps 15000 -ChunkSeconds 0 -MasksDir masks_refined -Init depth -DepthLoss 1.0 `
      -Composite -MaxAnisotropy 10 -AlphaOutsideWeight 1.0
.EXAMPLE
  # the same with the quality options (see prep\windows\README.md "Quality options"), camera 0004 held out
  ... -Stages init,train,export -Name q3s -FrameEnd 90 -Steps 15000 -ChunkSeconds 0 `
      -MasksDir masks_refined -Init depth -DepthLoss 1.0 -Composite -MaxAnisotropy 4 -AlphaOutsideWeight 1.0 `
      -DepthOffsets -DepthEdgePx 2 -LearnAppearance -HoldoutViews 3
#>
param(
    [Parameter(Mandatory = $true)][string]$Capture,
    [string]$Out = "",
    [ValidateSet("n", "l", "r", "u")][string]$Rotate = "n",
    [string[]]$Stages = @("inspect", "extract", "calibrate", "mask", "init", "train", "export", "asog"),
    [ValidateSet("best", "basic")][string]$Preset = "best",
    [string]$Name = "full",
    [int]$Steps = 30000,
    [int]$StepsPerFrame = 165,
    [string]$AsogRepo = "",
    [string[]]$AsogArgs = @(),
    [double]$ChunkSeconds = 3.0,
    [double]$MaxSigmaT = 0.5,
    [ValidateSet("cuda", "cpu")][string]$AsogDevice = "cuda",
    [int]$FrameStart = 1,
    [int]$FrameEnd = 0,
    [int]$ExtractStart = 0,
    [int]$ExtractFrames = 0,
    [string]$CalibrationFrom = "",
    [ValidateSet("turbo", "fast", "base", "precise")][string]$RomaSetting = "precise",
    [string]$EnvCmd = "C:\o4d\o4d-env.cmd",
    [string]$FromRun = "",
    [double]$PruneOpacity = 0.005,
    [ValidateSet("mixed", "motion32", "float16", "float32")][string]$TensorDtype = "mixed",
    [double]$PackChunkSeconds = 10.0,
    [switch]$NoPly,
    [switch]$CompressPly,
    [switch]$Force,
    [switch]$NoMasks,
    [double]$AlphaOutsideWeight,
    [double]$AlphaMaskWeight,
    [double]$MaxAnisotropy,
    [switch]$Composite,
    [string]$TrainExtra = "",
    [string]$MasksDir = "masks",
    [ValidateSet("roma", "depth")][string]$Init = "roma",
    [int]$DepthInitPoints = 60000,
    [double]$DepthLoss,
    [string]$TrainerSrc = "",
    [int[]]$HoldoutViews = @(),
    [switch]$LearnAppearance,
    [switch]$RandomBackground,
    [switch]$LearnTimeOffsets,
    [ValidateRange(0, 3)][int]$ShDegree = 3,
    [switch]$DepthOffsets,
    [int]$DepthEdgePx,
    [double]$DepthLossFinal,
    [int]$DepthDecayStart = 5000,
    [int]$DepthDecaySteps = 2000,
    [double]$DepthDistortion,
    [int]$DepthDistortionStart = 0,
    [double]$OpacityBinary,
    [int]$OpacityBinaryStart = 0
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Capture = (Resolve-Path $Capture).Path
if (-not $Out) { $Out = Join-Path $Capture "o4d" }
New-Item -ItemType Directory -Force $Out, (Join-Path $Out "logs") | Out-Null
$Out = (Resolve-Path $Out).Path
if (-not (Test-Path $EnvCmd)) { throw "$EnvCmd not found; run prep\windows\setup_env.ps1 first" }

function Run-Stage([string]$stage, [string]$done, [string]$cmdline, [string]$logName = "") {
    if ($Stages -notcontains $stage) { return }
    if ($done -and (Test-Path $done) -and -not $Force) { Write-Host "[$stage] skip (exists: $done)"; return }
    if (-not $logName) { $logName = $stage }
    $log = Join-Path $Out "logs\$logName.log"
    Write-Host "`n=== [$stage] $(Get-Date -Format 'HH:mm:ss') -> $log ===" -ForegroundColor Cyan
    $t = Get-Date
    & cmd.exe /d /c "call `"$EnvCmd`" >nul 2>&1 && cd /d `"$Out`" && $cmdline 2>&1" | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "[$stage] failed (exit $LASTEXITCODE); see $log" }
    Write-Host ("[{0}] done in {1:n1} min" -f $stage, ((Get-Date) - $t).TotalMinutes) -ForegroundColor Green
}

$Stages = @($Stages | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$known = "inspect", "extract", "calibrate", "mask", "refine", "consistent", "register", "offsets", "init", "train", "export", "asog"
$bad = $Stages | Where-Object { $known -notcontains $_ }
if ($bad) { throw "unknown stage(s): $bad (valid: $known)" }
$useMasks = -not $NoMasks

# -Preset best (default): the tested recipe, for every option not given on the command line
# (see prep\windows\README.md "Default recipe"). -Preset basic: the trainer defaults.
if ($Preset -eq "best") {
    $recipe = [ordered]@{ Init = "depth"; DepthLoss = 1.0; MaxAnisotropy = 10.0; DepthEdgePx = 2; DepthOffsets = $true }
    if ($useMasks) {
        $recipe["MasksDir"] = "masks_refined"; $recipe["Composite"] = $true; $recipe["AlphaOutsideWeight"] = 1.0
    }
    foreach ($k in $recipe.Keys) {
        if (-not $PSBoundParameters.ContainsKey($k)) {
            $v = $recipe[$k]
            if ($v -is [bool]) { $v = [switch]$v }
            Set-Variable -Name $k -Value $v
            $PSBoundParameters[$k] = $v
        }
    }
}
if ($NoMasks -and ($Stages -contains "mask")) {
    Write-Host "[mask] skipped (-NoMasks)"
    $Stages = @($Stages | Where-Object { $_ -ne "mask" })
}

$py = "python"
$x = "`"$Repo\prep\extract\extract_azure_kinect.py`""
$overwriteArg = if ($Force) { "--overwrite" } else { "" }
$sectionArg = ""
if ($ExtractStart -gt 0) { $sectionArg += " --start-frame $ExtractStart" }
if ($ExtractFrames -gt 0) { $sectionArg += " --num-frames $ExtractFrames" }

Run-Stage "inspect" (Join-Path $Out "k4a_report\inspect.json") `
    "$py $x inspect --input-dir `"$Capture`" --output-dir `"$Out`" --rotate $Rotate"
Run-Stage "extract" (Join-Path $Out "k4a_cameras.json") `
    "$py $x extract --input-dir `"$Capture`" --output-dir `"$Out`" --rotate $Rotate$sectionArg $overwriteArg"
if ($CalibrationFrom -and $Stages -contains "calibrate" -and ($Force -or -not (Test-Path (Join-Path $Out "reference_export.json")))) {
    # Reuse another take's calibration (same rig, cameras not moved): same serial per camera folder required.
    $src = (Resolve-Path $CalibrationFrom).Path
    $mine = (Get-Content (Join-Path $Out "k4a_cameras.json") -Raw | ConvertFrom-Json).cameras
    $theirs = (Get-Content (Join-Path $src "k4a_cameras.json") -Raw | ConvertFrom-Json).cameras
    $rename = @{}
    foreach ($p in $mine.PSObject.Properties) {
        $t = $theirs.($p.Name)
        if (-not $t -or $t.serial -ne $p.Value.serial) { throw "-CalibrationFrom: camera $($p.Name) serial differs from $src" }
        $rename[$t.source] = $p.Value.source
    }
    New-Item -ItemType Directory -Force (Join-Path $Out "sparse\0"), (Join-Path $Out "k4a_report") | Out-Null
    foreach ($f in "sparse\0\cameras.txt", "sparse\0\images.txt", "sparse\0\points3D.txt", "sparse\scene_meta.json",
                   "reference_export.json", "k4a_report\calibration.json", "k4a_report\calibration_points.npy") {
        $from = Join-Path $src $f
        if (-not (Test-Path $from)) { continue }
        if ($f -like "*.json") {
            $text = Get-Content $from -Raw
            foreach ($k in $rename.Keys) { $text = $text.Replace("`"$k`"", "`"$($rename[$k])`"") }
            [IO.File]::WriteAllText((Join-Path $Out $f), $text)
        } else { Copy-Item $from (Join-Path $Out $f) -Force }
    }
    Write-Host "[calibrate] copied from $src"
} elseif (-not $CalibrationFrom) {
    Run-Stage "calibrate" (Join-Path $Out "reference_export.json") `
        "$py `"$Repo\prep\align\calibrate_azure_kinect.py`" --data-dir . --overwrite"
}

$nFrames = 0
if (Test-Path (Join-Path $Out "k4a_cameras.json")) {
    $nFrames = [int]((Get-Content (Join-Path $Out "k4a_cameras.json") -Raw | ConvertFrom-Json).num_frames)
}
if ($nFrames -eq 0 -and -not ($Stages | Where-Object { $_ -ne "inspect" })) { exit 0 }  # inspect only
if ($FrameEnd -le 0) { $FrameEnd = $nFrames }
$fps = 30
if (Test-Path (Join-Path $Out "k4a_cameras.json")) { $fps = [double]((Get-Content (Join-Path $Out "k4a_cameras.json") -Raw | ConvertFrom-Json).fps) }
$count = $FrameEnd - $FrameStart + 1
if ($count -le 0) { throw "empty frame range $FrameStart..$FrameEnd" }
if ($nFrames -gt 0 -and $FrameEnd -gt $nFrames) { throw "-FrameEnd $FrameEnd exceeds the $nFrames extracted frames" }
# -Preset best: training steps scale with the clip (165 per frame = 15000 for 3 s, as tested),
# rounded to 1000, at least 5000.
if ($Preset -eq "best" -and -not $PSBoundParameters.ContainsKey("Steps")) {
    $Steps = [int]([math]::Max(5, [math]::Round($StepsPerFrame * $count / 1000.0)) * 1000)
}
Write-Host "Preset $Preset, $Steps training steps"

# A mask report written by a --qa-only tuning run (sample frames only) does not count as done.
function Test-ReportComplete([string]$report) {
    if (-not (Test-Path $report)) { return $false }
    return -not (Get-Content $report -Raw | ConvertFrom-Json).qa_only
}
$coarseReport = Join-Path $Out "mask_report\masks.json"
$coarseDone = ""
if (Test-ReportComplete $coarseReport) { $coarseDone = $coarseReport }
Run-Stage "mask" $coarseDone `
    "$py `"$Repo\prep\mask\depth_masks.py`" --data-dir ."

# -MasksDir masks_refined: colour refinement of the depth masks (prep\mask\refine_masks.py,
# needs SAM 2, see prep\mask\README-refine-masks.md); runs automatically when missing.
$refined = $MasksDir -ne "masks"
$consistent = $MasksDir -eq "masks_consistent"
$maskReport = $coarseReport
$needMasks = ($Stages -contains "init") -or ($Stages -contains "train")
if ($useMasks -and $refined) {
    if (@("masks_refined", "masks_consistent") -notcontains $MasksDir) {
        throw "-MasksDir must be masks, masks_refined or masks_consistent"
    }
    $maskReport = Join-Path $Out "mask_report\refine.json"
    $refineDone = ""
    if (Test-ReportComplete $maskReport) { $refineDone = $maskReport }
    if (-not $refineDone -and $needMasks) { $Stages += "refine" }
    Run-Stage "refine" $refineDone "$py `"$Repo\prep\mask\refine_masks.py`" --data-dir ."
}
# -MasksDir masks_consistent: cross-view vote over the refined masks (prep\mask\consistent_masks.py);
# pixels the views disagree on are marked unknown (mask PNG alpha) and left out of the mask losses.
if ($useMasks -and $consistent) {
    $refineReport = $maskReport
    $maskReport = Join-Path $Out "mask_report\consistent.json"
    $consDone = ""
    # Done when the report covers the frame range and is newer than the refined masks.
    if ((Test-ReportComplete $maskReport) -and -not $Force) {
        # "frames": the list, or [first, last, count] for long runs
        $consFrames = @((Get-Content $maskReport -Raw | ConvertFrom-Json).frames)
        $consLast = ($consFrames | Measure-Object -Maximum).Maximum
        $older = (Get-Item $maskReport).LastWriteTime -lt (Get-Item $refineReport).LastWriteTime
        if (-not $older -and $consFrames[0] -le $FrameStart -and $consLast -ge $FrameEnd) { $consDone = $maskReport }
    }
    if (-not $consDone -and $needMasks) { $Stages += "consistent" }
    Run-Stage "consistent" $consDone ("$py `"$Repo\prep\mask\consistent_masks.py`" --data-dir . --unknown " +
        "--frame-start $FrameStart --frame-end $FrameEnd")
}
function Test-MasksComplete { return (Test-ReportComplete $maskReport) }

# -DepthLoss: colour-registered depth maps (the trainer's --depth-root contract).
if ($PSBoundParameters.ContainsKey("DepthLoss") -and ($Stages -contains "train")) {
    $regDone = Join-Path $Out "depth_registered\_complete.json"
    if (-not (Test-Path $regDone)) { $Stages += "register" }
    Run-Stage "register" $regDone "$py `"$Repo\prep\extract\k4a_register_depth.py`" --data-dir ."
}

$initArgs = ""
$trainArgs = ""
$npzStem = "keyframes"
$fgDir = "foregrounds"
if ($useMasks) {
    if ((($Stages -contains "init") -or ($Stages -contains "train")) -and -not (Test-MasksComplete)) {
        throw "no complete mask set ($maskReport missing or from --qa-only): run the mask stage, or pass -NoMasks"
    }
    $npzStem = if ($consistent) { "keyframes_consistent" } elseif ($refined) { "keyframes_refined" } else { "keyframes_masked" }
    if ($refined) { $fgDir = if ($consistent) { "foregrounds_consistent" } else { "foregrounds_refined" } }
    $initArgs = " --masks-root $MasksDir"
    $trainArgs = " --masks-root $MasksDir --config.masked-rgb-loss"
    if ($PSBoundParameters.ContainsKey("AlphaOutsideWeight")) { $trainArgs += " --config.lambda-alpha-outside-mask $AlphaOutsideWeight" }
    if ($PSBoundParameters.ContainsKey("AlphaMaskWeight")) { $trainArgs += " --config.lambda-alpha-mask $AlphaMaskWeight" }
    if ($Composite) {
        # Composite the render over a white plate and supervise the whole frame (restores the
        # photometric penalty on splats at the wrong depth); needs straight foreground RGB.
        $trainArgs += " --config.masked-rgb-composite --foregrounds-root $fgDir"
    }
} elseif ($PSBoundParameters.ContainsKey("AlphaOutsideWeight") -or $PSBoundParameters.ContainsKey("AlphaMaskWeight")) {
    Write-Host "-AlphaOutsideWeight/-AlphaMaskWeight ignored with -NoMasks" -ForegroundColor Yellow
}
if ($Composite -and -not $useMasks) { throw "-Composite needs masks; drop -NoMasks" }
if ($PSBoundParameters.ContainsKey("MaxAnisotropy")) { $trainArgs += " --config.max-scale-anisotropy $MaxAnisotropy" }
if ($Init -eq "depth") {
    $npzStem = $npzStem -replace "^keyframes", "keyframes_depth"
    if (-not $useMasks) { $initArgs = " --masks-root none" }
    # the trainer subsamples every keyframe to target-count-per-keyframe (default 25,862)
    $trainArgs += " --config.target-count-per-keyframe $DepthInitPoints"
}
if ($PSBoundParameters.ContainsKey("DepthLoss")) {
    $trainArgs += " --depth-root depth_registered --config.lambda-depth $DepthLoss"
    if ($PSBoundParameters.ContainsKey("DepthEdgePx")) { $trainArgs += " --config.depth-edge-px $DepthEdgePx" }
    if ($PSBoundParameters.ContainsKey("DepthLossFinal")) {
        # constant lambda until -DepthDecayStart, then log-linear to lambda x -DepthLossFinal
        $trainArgs += " --config.depth-loss-final-scale $DepthLossFinal --config.depth-loss-decay-start-step $DepthDecayStart" +
            " --config.depth-loss-decay-steps $DepthDecaySteps"
    }
} elseif ($PSBoundParameters.ContainsKey("DepthEdgePx") -or $PSBoundParameters.ContainsKey("DepthLossFinal")) {
    throw "-DepthEdgePx/-DepthLossFinal need -DepthLoss"
}
# -DepthOffsets: per-camera additive depth corrections (prep\extract\k4a_depth_offsets.py ->
# depth_offsets.json), applied to the depth init and the depth loss targets.
if ($DepthOffsets) {
    if (-not ($PSBoundParameters.ContainsKey("DepthLoss") -or $Init -eq "depth")) { throw "-DepthOffsets needs -DepthLoss or -Init depth" }
    $offsDone = Join-Path $Out "depth_offsets.json"
    if (-not (Test-Path $offsDone) -and (($Stages -contains "init") -or ($Stages -contains "train"))) { $Stages += "offsets" }
    $offsMasks = if ($useMasks) { "--masks-dir $MasksDir" } else { "--masks-dir masks" }
    Run-Stage "offsets" $offsDone "$py `"$Repo\prep\extract\k4a_depth_offsets.py`" --data-dir . $offsMasks"
    if ($PSBoundParameters.ContainsKey("DepthLoss")) { $trainArgs += " --depth-offsets-json depth_offsets.json" }
    if ($Init -eq "depth") { $initArgs += " --depth-offsets-json depth_offsets.json"; $npzStem += "_doff" }
}
if ($HoldoutViews.Count -gt 0) {
    # 0-based positions in the sorted camera list (0 = first camera folder); psnr/holdout_* in TensorBoard
    $trainArgs += " --config.holdout-views " + (($HoldoutViews | ForEach-Object { [string]$_ }) -join " ")
}
if ($LearnAppearance) { $trainArgs += " --config.learn-camera-appearance" }
if ($RandomBackground) {
    if (-not $Composite) { throw "-RandomBackground needs -Composite" }
    $trainArgs += " --config.masked-rgb-random-bg"
}
if ($LearnTimeOffsets) { $trainArgs += " --config.learn-camera-time-offsets" }
if ($ShDegree -ne 3) { $trainArgs += " --sh-degree $ShDegree" }
if ($PSBoundParameters.ContainsKey("DepthDistortion")) {
    $trainArgs += " --config.lambda-depth-distortion $DepthDistortion --config.depth-distortion-start-step $DepthDistortionStart"
}
if ($PSBoundParameters.ContainsKey("OpacityBinary")) {
    $trainArgs += " --config.lambda-opacity-binary $OpacityBinary --config.opacity-binary-start-step $OpacityBinaryStart"
}
# -TrainerSrc: run train/export from another trainer checkout's src folder instead of the
# installed trainer (e.g. to try a trainer branch without reinstalling).
$trainerEnv = ""
if ($TrainerSrc) {
    if (-not (Test-Path (Join-Path $TrainerSrc "trigs"))) { throw "-TrainerSrc $TrainerSrc has no trigs package" }
    $trainerEnv = "set `"PYTHONPATH=$TrainerSrc`" && "
}
# Long clips: the trainer's per-image caches (300 entries each for foregrounds, masks and depth)
# hit rarely when frames are drawn at random from thousands of images, but hold several GB of
# host RAM next to the CPU copy of every splat; a 377-frame run ran out of 32 GB with them.
if ($count -gt 100 -and $TrainExtra -notmatch "cache-size") {
    $trainArgs += " --mask-cache-size 48 --depth-cache-size 48 --foreground-cache-size 48"
}
if ($TrainExtra) { $trainArgs += " $TrainExtra" }

# The frame range is part of the keyframe file name: a 3 s test and a full-clip run must not
# share (and silently skip) each other's initialisation.
$npzFile = "{0}_f{1:d4}-{2:d4}.npz" -f $npzStem, $FrameStart, $FrameEnd
Write-Host "Frames $FrameStart..$FrameEnd ($count frames, $([math]::Round($count / $fps, 2)) s) -> $npzFile"

if ($Composite -and (($Stages -contains "train") -or ($Stages -contains "init"))) {
    # Straight foregrounds for composite training; rebuilt when missing or older than the masks.
    $fgDone = Join-Path $Out "$fgDir\_complete.json"
    $fgStale = (-not (Test-Path $fgDone)) -or ((Get-Item $fgDone).LastWriteTime -lt (Get-Item $maskReport).LastWriteTime)
    if (-not $fgStale) {
        $fgRange = (Get-Content $fgDone -Raw | ConvertFrom-Json).frames
        $fgStale = ($fgRange[0] -gt $FrameStart) -or ($fgRange[1] -lt $FrameEnd)
    }
    if ($fgStale -or $Force) {
        $script:Stages += "foregrounds"
        Run-Stage "foregrounds" "" ("$py `"$Repo\prep\mask\straight_foregrounds.py`" --data-dir . " +
            "--masks-dir $MasksDir --output-dir $fgDir")
    } else { Write-Host "[foregrounds] skip (exists: $fgDone)" }
}

$npzStemName = [System.IO.Path]::GetFileNameWithoutExtension($npzFile)
if ($Init -eq "depth") {
    Run-Stage "init" (Join-Path $Out $npzFile) `
        ("$py `"$Repo\prep\init\depth_init.py`" --data-dir . --frame-start $FrameStart --frame-end $FrameEnd " +
         "--fps $fps --target-points $DepthInitPoints --output-path $npzFile$initArgs") "init-$npzStemName"
} else {
    Run-Stage "init" (Join-Path $Out $npzFile) `
        ("$py `"$Repo\prep\init\prep-roma-trigs-memory.py`" --reference-json reference_export.json --images-root images " +
         "--fps $fps --sync-json sync.json --frame-start $FrameStart --frame-end $FrameEnd " +
         "--roma-v2-setting $RomaSetting --output-path $npzFile$initArgs") "init-$npzStemName"
}

$chunk = ""
if ($ChunkSeconds -gt 0) { $chunk = "--config.temporal-chunk-len-s $ChunkSeconds --config.temporal-chunk-visit-steps 1000" }
# Clips longer than one chunk window: cap splat lifetimes (temporal sigma). Uncapped, a row's
# lifetime grows for free while only its own window is trained, and it then hangs, frozen, in
# other windows' frames (haze in front of a moving head; full 377-frame test capture). 0.5 s removed it.
if ($ChunkSeconds -gt 0 -and $MaxSigmaT -gt 0 -and ($count / $fps) -gt $ChunkSeconds -and
    $TrainExtra -notmatch "max-sigma-t") {
    $chunk += " --config.max-sigma-t $MaxSigmaT"
}
Run-Stage "train" (Join-Path $Out "runs\$Name\trigs_ckpt.pt") `
    "${trainerEnv}trigs-train-colmap --data-dir . --output-dir runs/$Name --npz-path $npzFile --config.n-steps $Steps $chunk$trainArgs" `
    "train-$Name"

# Export decodes a copy of the checkpoint (deliver\prepare_export_checkpoint.py): it carries
# the run's training config, so the PLYs and the .trigsbin use the temporal threshold the run
# was trained with, and leaves out rows below -PruneOpacity. trigs_ckpt.pt is never modified.
# -FromRun exports another run's checkpoint into runs\<Name> without training.
$srcRun = if ($FromRun) { $FromRun } else { $Name }
if ($FromRun -and ($Stages -contains "train")) { throw "-FromRun exports an existing run; drop train from -Stages" }
if (($Stages -contains "export") -and -not ($Stages -contains "train") -and
    -not (Test-Path (Join-Path $Out "runs\$srcRun\trigs_ckpt.pt"))) {
    throw "no checkpoint runs\$srcRun\trigs_ckpt.pt to export"
}
if ($PruneOpacity -lt 0 -or $PruneOpacity -ge 1) { throw "-PruneOpacity must be in [0, 1)" }
if ($PackChunkSeconds -le 0) { throw "-PackChunkSeconds must be > 0" }
$exportCkpt = "runs\$Name\export_ckpt.pt"
$exportCmd = "$py `"$Repo\deliver\prepare_export_checkpoint.py`" --checkpoint runs\$srcRun\trigs_ckpt.pt " +
    "--output $exportCkpt --min-opacity $PruneOpacity --n-frames $count --fps $fps --overwrite && "
# Frame count and fps come from the export copy (the run's own values).
if ($NoPly -and $CompressPly) { throw "-CompressPly needs the PLYs; drop -NoPly" }
if (-not $NoPly) { $exportCmd += "${trainerEnv}trigs-export-plys --ckpt-path $exportCkpt --output-dir runs\$Name\plys && " }
if ($CompressPly) {
    $exportCmd += "$py `"$Repo\deliver\compress_plys.py`" --input-dir runs\$Name\plys --delete-source && "
}
$exportCmd += "$py `"$Repo\deliver\make-trigs-timeline.py`" from-checkpoint --output runs\$Name\$Name.trigsbin " +
    "--checkpoint $exportCkpt --frame-rate $fps --max-chunk-duration-seconds $PackChunkSeconds " +
    "--chunk-overlap-seconds 0 --tensor-dtype $TensorDtype --overwrite --fade-frames 0 && " +
    "del `"$exportCkpt`""
Run-Stage "export" (Join-Path $Out "runs\$Name\$Name.trigsbin") $exportCmd "export-$Name"

# asog: Animated SOG (.asog) for the three.js / PlayCanvas players (arrival-space/animated-sog).
# The .trigsbin of the export stage is +Y up with the trainer's wxyz quaternions, which
# make-trigs-timeline.py labels xyzw; the players expect y down. This stage takes animated-sog's
# own route: `asog flipbook flip` turns a fresh export copy 180 deg about z and writes xyzw
# quaternions, make-trigs-timeline.py packs it to runs\<Name>\<Name>_ydown.trigsbin, and
# `asog encode` needs no extra options. The SH palette k-means runs on the GPU when there is one
# (-AsogDevice cpu hides CUDA, e.g. while another job trains). The encoder is the asog package
# of the environment, or a checkout: -AsogRepo, else animated-sog-fork or animated-sog next to
# this repo when present.
if ($Stages -contains "asog") {
    if (-not $AsogRepo) {
        foreach ($d in "animated-sog-fork", "animated-sog") {
            $c = Join-Path (Split-Path -Parent $Repo) $d
            if (Test-Path (Join-Path $c "python\src\asog")) { $AsogRepo = $c; break }
        }
    }
    $asogEnv = ""
    $encodeOpts = ""
    if ($AsogRepo) {
        if (-not (Test-Path (Join-Path $AsogRepo "python\src\asog"))) { throw "no animated-sog checkout at $AsogRepo (-AsogRepo)" }
        $AsogRepo = (Resolve-Path $AsogRepo).Path   # stages run from $Out
        $asogEnv = "set `"PYTHONPATH=$AsogRepo\python\src`" && "
        # --palette-device where the encoder has it (CPU k-means without the size limit)
        if (Select-String -Path (Join-Path $AsogRepo "python\src\asog\cli.py") -Pattern "palette-device" -Quiet) {
            $encodeOpts = " --palette-device $AsogDevice"
        }
    }
    if ($AsogDevice -eq "cpu") { $asogEnv = "set `"CUDA_VISIBLE_DEVICES=-1`" && " + $asogEnv }
    if (-not (Test-Path (Join-Path $Out "runs\$srcRun\trigs_ckpt.pt"))) { throw "no checkpoint runs\$srcRun\trigs_ckpt.pt to convert" }
    $extra = ($AsogArgs | ForEach-Object { $_ }) -join " "
    $asogCkpt = "runs\$Name\asog_ckpt.pt"
    $ydownCkpt = "runs\$Name\asog_ckpt_ydown.pt"
    $ydownBin = "runs\$Name\${Name}_ydown.trigsbin"
    $asogFile = "runs\$Name\$Name.asog"
    Run-Stage "asog" (Join-Path $Out $asogFile) (
        "$py `"$Repo\deliver\prepare_export_checkpoint.py`" --checkpoint runs\$srcRun\trigs_ckpt.pt " +
        "--output $asogCkpt --min-opacity $PruneOpacity --n-frames $count --fps $fps --overwrite && " +
        "$asogEnv$py -m asog.cli flipbook flip $asogCkpt $ydownCkpt 2>&1 && " +
        "$py `"$Repo\deliver\make-trigs-timeline.py`" from-checkpoint --output $ydownBin " +
        "--checkpoint $ydownCkpt --frame-rate $fps --max-chunk-duration-seconds $PackChunkSeconds " +
        "--chunk-overlap-seconds 0 --tensor-dtype $TensorDtype --overwrite --fade-frames 0 && " +
        "del `"$asogCkpt`" `"$ydownCkpt`" && " +
        "$asogEnv$py -m asog.cli encode $ydownBin $asogFile$encodeOpts $extra 2>&1 && " +
        "$asogEnv$py -m asog.cli info $asogFile") "asog-$Name"
    $root = Split-Path -Parent $Repo
    if ($Out.StartsWith($root)) {
        $player = if ($AsogRepo -and (Split-Path -Parent $AsogRepo) -eq $root) { Split-Path -Leaf $AsogRepo } else { "animated-sog" }
        $rel = (Join-Path $Out $asogFile).Substring($root.Length).TrimStart('\').Replace('\', '/')
        Write-Host "Player: powershell -File prep\windows\asog_convert.ps1 -SkipEncode -Serve, then open" -ForegroundColor Green
        Write-Host "  http://localhost:8765/$player/examples/three/?src=/$rel" -ForegroundColor Green
    }
}

Write-Host "`nOutputs in $Out" -ForegroundColor Green
