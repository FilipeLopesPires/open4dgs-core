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

  Added automatically when an option needs them and their output is missing:
    refine       -MasksDir masks_refined   masks shrunk to the colour edge   masks_refined\, mask_report\refine.json
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
  -DepthLoss needs a trainer with depth supervision (open4dgs-trigs-fork; see prep\windows\README.md).

  Logs: logs\<stage>.log, except init-<keyframe file>.log, train-<Name>.log, export-<Name>.log.

  Review k4a_report\*.jpg and the console QA lines after inspect/calibrate, and
  mask_report\mask_sheet.jpg after mask, before spending hours on init/train
  (see prep\extract\README-azure-kinect.md).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File prep\windows\run_kinect_pipeline.ps1 -Capture datasets\professor -Rotate l
.EXAMPLE
  # stop after the masks to review mask_report\mask_sheet.jpg
  ... -Capture datasets\professor -Rotate l -Stages inspect,extract,calibrate,mask
.EXAMPLE
  # unmasked run next to the masked default run "full"
  ... -Capture datasets\professor -Rotate l -NoMasks -Name nomask
.EXAMPLE
  # only re-train and export, 45k steps, stronger opacity penalty outside the mask, new run name
  ... -Capture datasets\professor -Rotate l -Stages train,export -Name long -Steps 45000 -AlphaOutsideWeight 1.0
.EXAMPLE
  # best tested quality recipe (3 s): refined masks, depth init, depth loss, composite, anisotropy cap
  ... -Capture datasets\professor -Rotate l -Stages init,train,export -Name all3s -FrameEnd 90 `
      -Steps 15000 -ChunkSeconds 0 -MasksDir masks_refined -Init depth -DepthLoss 1.0 `
      -Composite -MaxAnisotropy 10 -AlphaOutsideWeight 1.0
#>
param(
    [Parameter(Mandatory = $true)][string]$Capture,
    [string]$Out = "",
    [ValidateSet("n", "l", "r", "u")][string]$Rotate = "n",
    [string[]]$Stages = @("inspect", "extract", "calibrate", "mask", "init", "train", "export"),
    [string]$Name = "full",
    [int]$Steps = 30000,
    [double]$ChunkSeconds = 3.0,
    [int]$FrameStart = 1,
    [int]$FrameEnd = 0,
    [ValidateSet("turbo", "fast", "base", "precise")][string]$RomaSetting = "precise",
    [string]$EnvCmd = "C:\o4d\o4d-env.cmd",
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
    [string]$TrainerSrc = ""
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
$known = "inspect", "extract", "calibrate", "mask", "refine", "register", "init", "train", "export"
$bad = $Stages | Where-Object { $known -notcontains $_ }
if ($bad) { throw "unknown stage(s): $bad (valid: $known)" }
$useMasks = -not $NoMasks
if ($NoMasks -and ($Stages -contains "mask")) {
    Write-Host "[mask] skipped (-NoMasks)"
    $Stages = @($Stages | Where-Object { $_ -ne "mask" })
}

$py = "python"
$x = "`"$Repo\prep\extract\extract_azure_kinect.py`""
$overwriteArg = if ($Force) { "--overwrite" } else { "" }

Run-Stage "inspect" (Join-Path $Out "k4a_report\inspect.json") `
    "$py $x inspect --input-dir `"$Capture`" --output-dir `"$Out`" --rotate $Rotate"
Run-Stage "extract" (Join-Path $Out "k4a_cameras.json") `
    "$py $x extract --input-dir `"$Capture`" --output-dir `"$Out`" --rotate $Rotate $overwriteArg"
Run-Stage "calibrate" (Join-Path $Out "reference_export.json") `
    "$py `"$Repo\prep\align\calibrate_azure_kinect.py`" --data-dir . --overwrite"

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
$maskReport = $coarseReport
if ($useMasks -and $refined) {
    if ($MasksDir -ne "masks_refined") { throw "-MasksDir must be masks or masks_refined" }
    $maskReport = Join-Path $Out "mask_report\refine.json"
    $refineDone = ""
    if (Test-ReportComplete $maskReport) { $refineDone = $maskReport }
    if (-not $refineDone -and (($Stages -contains "init") -or ($Stages -contains "train"))) { $Stages += "refine" }
    Run-Stage "refine" $refineDone "$py `"$Repo\prep\mask\refine_masks.py`" --data-dir ."
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
    $npzStem = if ($refined) { "keyframes_refined" } else { "keyframes_masked" }
    if ($refined) { $fgDir = "foregrounds_refined" }
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
}
# -TrainerSrc: run train/export from another trainer checkout's src folder instead of the
# installed trainer (e.g. to try a trainer branch without reinstalling).
$trainerEnv = ""
if ($TrainerSrc) {
    if (-not (Test-Path (Join-Path $TrainerSrc "trigs"))) { throw "-TrainerSrc $TrainerSrc has no trigs package" }
    $trainerEnv = "set `"PYTHONPATH=$TrainerSrc`" && "
}
if ($TrainExtra) { $trainArgs += " $TrainExtra" }

$nFrames = 0
if (Test-Path (Join-Path $Out "k4a_cameras.json")) {
    $nFrames = [int]((Get-Content (Join-Path $Out "k4a_cameras.json") -Raw | ConvertFrom-Json).num_frames)
}
if ($FrameEnd -le 0) { $FrameEnd = $nFrames }
$fps = 30
if (Test-Path (Join-Path $Out "k4a_cameras.json")) { $fps = [double]((Get-Content (Join-Path $Out "k4a_cameras.json") -Raw | ConvertFrom-Json).fps) }
$count = $FrameEnd - $FrameStart + 1
if ($count -le 0) { throw "empty frame range $FrameStart..$FrameEnd" }
if ($nFrames -gt 0 -and $FrameEnd -gt $nFrames) { throw "-FrameEnd $FrameEnd exceeds the $nFrames extracted frames" }
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
Run-Stage "train" (Join-Path $Out "runs\$Name\trigs_ckpt.pt") `
    "${trainerEnv}trigs-train-colmap --data-dir . --output-dir runs/$Name --npz-path $npzFile --config.n-steps $Steps $chunk$trainArgs" `
    "train-$Name"

Run-Stage "export" (Join-Path $Out "runs\$Name\$Name.trigsbin") `
    ("${trainerEnv}trigs-export-plys --ckpt-path runs/$Name/trigs_ckpt.pt --output-dir runs/$Name/plys --n-frames $count --fps $fps && " +
     "$py `"$Repo\deliver\make-trigs-timeline.py`" from-checkpoint --output runs/$Name/$Name.trigsbin " +
     "--checkpoint runs/$Name/trigs_ckpt.pt --frame-rate $fps --total-frames $count " +
     "--chunk-duration-seconds 2.5 --chunk-overlap-seconds 0 --tensor-dtype motion32 --overwrite --fade-frames 0") `
    "export-$Name"

Write-Host "`nOutputs in $Out" -ForegroundColor Green
