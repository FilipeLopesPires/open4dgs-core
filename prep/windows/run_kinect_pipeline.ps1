<#
.SYNOPSIS
  Azure Kinect capture -> 4DGS, end to end on Windows (after setup_env.ps1).

.DESCRIPTION
  Stages (each skipped when its output exists, unless -Force; logs go to <Out>\logs\):
    inspect    audit, frame sync, QA sheets               k4a_report\inspect.json
    extract    undistorted frames + depth + sync.json      images\, depth\, sync.json
    calibrate  extrinsics from colour + depth              sparse\0, reference_export.json
    mask       foreground masks from multi-view depth      masks\, mask_report\masks.json
    init       RoMa v2 + MEMFOF point initialisation       keyframes_masked.npz (keyframes.npz with -NoMasks)
    train      TriGS training (time-chunked by default)    runs\<Name>\trigs_ckpt.pt
    export     per-frame PLYs + .trigsbin timeline         runs\<Name>\plys\, runs\<Name>\<Name>.trigsbin

  Masks are on by default (prep\mask\depth_masks.py, see prep\mask\README-depth-masks.md):
  init drops background points (--masks-root masks) and train uses the masked RGB loss
  (--masks-root masks --config.masked-rgb-loss), which pushes opacity outside the mask to 0.
  -NoMasks skips the mask stage and runs init/train without masks (the previous behaviour).
  Masked and unmasked runs use different keyframe files, but the same run folder
  runs\<Name>: give each its own -Name to keep both.

  -AlphaOutsideWeight / -AlphaMaskWeight set --config.lambda-alpha-outside-mask /
  --config.lambda-alpha-mask; when omitted the trainer defaults apply.

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
    [double]$AlphaMaskWeight
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Capture = (Resolve-Path $Capture).Path
if (-not $Out) { $Out = Join-Path $Capture "o4d" }
New-Item -ItemType Directory -Force $Out, (Join-Path $Out "logs") | Out-Null
$Out = (Resolve-Path $Out).Path
if (-not (Test-Path $EnvCmd)) { throw "$EnvCmd not found; run prep\windows\setup_env.ps1 first" }

function Run-Stage([string]$stage, [string]$done, [string]$cmdline) {
    if ($Stages -notcontains $stage) { return }
    if ($done -and (Test-Path $done) -and -not $Force) { Write-Host "[$stage] skip (exists: $done)"; return }
    $log = Join-Path $Out "logs\$stage.log"
    Write-Host "`n=== [$stage] $(Get-Date -Format 'HH:mm:ss') -> $log ===" -ForegroundColor Cyan
    $t = Get-Date
    & cmd.exe /d /c "call `"$EnvCmd`" >nul 2>&1 && cd /d `"$Out`" && $cmdline 2>&1" | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "[$stage] failed (exit $LASTEXITCODE); see $log" }
    Write-Host ("[{0}] done in {1:n1} min" -f $stage, ((Get-Date) - $t).TotalMinutes) -ForegroundColor Green
}

$Stages = @($Stages | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$known = "inspect", "extract", "calibrate", "mask", "init", "train", "export"
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

# masks.json is also written by a --qa-only tuning run (sample frames only); that does not count as done.
$maskReport = Join-Path $Out "mask_report\masks.json"
function Test-MasksComplete {
    if (-not (Test-Path $maskReport)) { return $false }
    return -not (Get-Content $maskReport -Raw | ConvertFrom-Json).qa_only
}
$maskDone = ""
if (Test-MasksComplete) { $maskDone = $maskReport }
Run-Stage "mask" $maskDone `
    "$py `"$Repo\prep\mask\depth_masks.py`" --data-dir ."

$initArgs = ""
$trainArgs = ""
$npzFile = "keyframes.npz"
if ($useMasks) {
    if ((($Stages -contains "init") -or ($Stages -contains "train")) -and -not (Test-MasksComplete)) {
        throw "no complete mask set ($maskReport missing or from --qa-only): run the mask stage, or pass -NoMasks"
    }
    $npzFile = "keyframes_masked.npz"
    $initArgs = " --masks-root masks"
    $trainArgs = " --masks-root masks --config.masked-rgb-loss"
    if ($PSBoundParameters.ContainsKey("AlphaOutsideWeight")) { $trainArgs += " --config.lambda-alpha-outside-mask $AlphaOutsideWeight" }
    if ($PSBoundParameters.ContainsKey("AlphaMaskWeight")) { $trainArgs += " --config.lambda-alpha-mask $AlphaMaskWeight" }
} elseif ($PSBoundParameters.ContainsKey("AlphaOutsideWeight") -or $PSBoundParameters.ContainsKey("AlphaMaskWeight")) {
    Write-Host "-AlphaOutsideWeight/-AlphaMaskWeight ignored with -NoMasks" -ForegroundColor Yellow
}

$nFrames = 0
if (Test-Path (Join-Path $Out "k4a_cameras.json")) {
    $nFrames = [int]((Get-Content (Join-Path $Out "k4a_cameras.json") -Raw | ConvertFrom-Json).num_frames)
}
if ($FrameEnd -le 0) { $FrameEnd = $nFrames }
$fps = 30
if (Test-Path (Join-Path $Out "k4a_cameras.json")) { $fps = [double]((Get-Content (Join-Path $Out "k4a_cameras.json") -Raw | ConvertFrom-Json).fps) }
$count = $FrameEnd - $FrameStart + 1

Run-Stage "init" (Join-Path $Out $npzFile) `
    ("$py `"$Repo\prep\init\prep-roma-trigs-memory.py`" --reference-json reference_export.json --images-root images " +
     "--fps $fps --sync-json sync.json --frame-start $FrameStart --frame-end $FrameEnd " +
     "--roma-v2-setting $RomaSetting --output-path $npzFile$initArgs")

$chunk = ""
if ($ChunkSeconds -gt 0) { $chunk = "--config.temporal-chunk-len-s $ChunkSeconds --config.temporal-chunk-visit-steps 1000" }
Run-Stage "train" (Join-Path $Out "runs\$Name\trigs_ckpt.pt") `
    "trigs-train-colmap --data-dir . --output-dir runs/$Name --npz-path $npzFile --config.n-steps $Steps $chunk$trainArgs"

Run-Stage "export" (Join-Path $Out "runs\$Name\$Name.trigsbin") `
    ("trigs-export-plys --ckpt-path runs/$Name/trigs_ckpt.pt --output-dir runs/$Name/plys --n-frames $count --fps $fps && " +
     "$py `"$Repo\deliver\make-trigs-timeline.py`" from-checkpoint --output runs/$Name/$Name.trigsbin " +
     "--checkpoint runs/$Name/trigs_ckpt.pt --frame-rate $fps --total-frames $count " +
     "--chunk-duration-seconds 2.5 --chunk-overlap-seconds 0 --tensor-dtype motion32 --overwrite --fade-frames 0")

Write-Host "`nOutputs in $Out" -ForegroundColor Green
