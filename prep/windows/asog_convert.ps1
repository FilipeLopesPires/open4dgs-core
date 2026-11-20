<#
.SYNOPSIS
  Convert an Open4DGS TRiGS export (.trigsbin) to Animated SOG (.asog) with the animated-sog fork, and
  serve the fork's three.js player on it.

.DESCRIPTION
  Encoder: animated-sog-fork\python (run from source with PYTHONPATH, inside C:\o4d\venv: it only needs
  numpy + torch, both already there; nothing is installed). The SH palette k-means runs on the CPU
  (torch, --palette-device cpu) and CUDA is hidden, so a training job on the GPU is not disturbed.

  Our .trigsbin files need three encoder options (added in the fork):
    --quat-order wxyz   make-trigs-timeline.py copies the trainer's wxyz quaternions but labels them "xyzw"
    --transform z180    our world is +Y up (floor at y = 0); .asog is y down (players turn it 180 deg about z)
    (temporal threshold) read from the file (semantics.temporalCullThreshold = 0.01) and kept in the .asog
                         as motion.threshold; the players honour it (default 0.05 only when absent)

  Encoder quality / size options are left at their defaults (qp 2, qa 8, prune 0.3, mbits 14, tbits 11,
  lsbits 8, shk 65536, level 9); pass more with -EncodeArgs, e.g. -EncodeArgs '--prune','0'.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File prep\windows\asog_convert.ps1                     # all3s_compressed
  powershell -ExecutionPolicy Bypass -File prep\windows\asog_convert.ps1 -Run all3s_fixed
  powershell -ExecutionPolicy Bypass -File prep\windows\asog_convert.ps1 -SkipEncode -Serve  # player only
#>
param(
    [string]$Run = "all3s_compressed",        # run folder under datasets\professor\o4d\runs, holding <Run>.trigsbin
    [string]$Trigsbin = "",                   # explicit input (default: runs\<Run>\<Run>.trigsbin)
    [string]$Out = "",                        # explicit output (default: runs\asog\<Run>.asog)
    [string[]]$EncodeArgs = @(),              # extra `asog encode` options
    [switch]$SkipEncode,                      # only (re)serve
    [switch]$Check,                           # also run `asog check` (error report against the .trigsbin, ~1-2 min)
    [switch]$Serve,                           # start a static server for the player (in this window)
    [int]$Port = 8765
)
$ErrorActionPreference = "Stop"

$Core = (Resolve-Path "$PSScriptRoot\..\..").Path                        # open4dgs-core-fork
$Root = Split-Path $Core -Parent                                          # Open4DGS (both repos)
$Fork = Join-Path $Root "animated-sog-fork"
$Runs = Join-Path $Core "datasets\professor\o4d\runs"
if (-not $Trigsbin) { $Trigsbin = Join-Path $Runs "$Run\$Run.trigsbin" }
if (-not $Out) { $Out = Join-Path $Runs "asog\$Run.asog" }
New-Item -ItemType Directory -Force (Split-Path $Out -Parent) | Out-Null

function Invoke-Asog([string[]]$AsogArgs) {
    $quoted = ($AsogArgs | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join " "
    # CUDA hidden: CPU only. PYTHONPATH: the fork's encoder from source.
    $line = "call C:\o4d\o4d-env.cmd >nul 2>nul && set `"CUDA_VISIBLE_DEVICES=-1`" && set `"PYTHONPATH=$Fork\python\src`" && python -m asog.cli $quoted 2>&1"   # log on stdout (PowerShell 5.1 turns stderr lines into errors)
    cmd /c $line
    if ($LASTEXITCODE -ne 0) { throw "asog $($AsogArgs[0]) failed ($LASTEXITCODE)" }
}

if (-not $SkipEncode) {
    if (-not (Test-Path $Trigsbin)) { throw "no input $Trigsbin" }
    $t0 = Get-Date
    Invoke-Asog (@("encode", $Trigsbin, $Out, "--quat-order", "wxyz", "--transform", "z180", "--palette-device", "cpu") + $EncodeArgs)
    "encode: {0:n0} s" -f ((Get-Date) - $t0).TotalSeconds
    Invoke-Asog @("info", $Out)
    "{0}: {1:n1} MB  <-  {2}: {3:n1} MB" -f $Out, ((Get-Item $Out).Length / 1MB), $Trigsbin, ((Get-Item $Trigsbin).Length / 1MB)
    if ($Check) { Invoke-Asog @("check", $Trigsbin, $Out) }
}

# the player: serve the Open4DGS folder (fork + data under one origin); three.js comes from the jsDelivr CDN
$rel = $Out.Substring($Root.Length).TrimStart('\').Replace('\', '/')
$url = "http://localhost:$Port/animated-sog-fork/examples/three/?src=/$rel"
"player: $url"
if ($Serve) {
    "serving $Root on port $Port (Ctrl+C stops)"
    & C:\o4d\venv\Scripts\python.exe -m http.server $Port --bind 127.0.0.1 --directory $Root
}
