<#
.SYNOPSIS
  Serve the animated-sog three.js player on local .asog files, and optionally encode a y-down
  .trigsbin to .asog first.

.DESCRIPTION
  run_kinect_pipeline.ps1 -Stages asog is the usual way to make an .asog: it turns the run's
  checkpoint y down (`asog flipbook flip`), packs <Name>_ydown.trigsbin and encodes it. This script
  re-encodes such a y-down .trigsbin with other encoder options (-EncodeArgs), and serves the
  player for any .asog under the folder that holds this repo and the animated-sog checkout.

  The encoder is the asog package of the environment (-EnvCmd), or a checkout: -AsogRepo, else
  animated-sog-fork or animated-sog next to this repo when present. Encoder options are left at
  their defaults; pass more with -EncodeArgs, e.g. -EncodeArgs '--prune','0'.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File prep\windows\asog_convert.ps1 -Serve        # player only
  powershell -ExecutionPolicy Bypass -File prep\windows\asog_convert.ps1 `
      -Trigsbin <capture>\o4d\runs\<Name>\<Name>_ydown.trigsbin -Out <capture>\o4d\runs\<Name>\<Name>_p0.asog `
      -EncodeArgs '--prune','0' -Check -Serve
#>
param(
    [string]$Trigsbin = "",                   # y-down .trigsbin to encode (none: only serve)
    [string]$Out = "",                        # output (default: next to the .trigsbin, .asog)
    [string[]]$EncodeArgs = @(),              # extra `asog encode` options
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",   # cpu hides CUDA (e.g. while a job trains)
    [string]$AsogRepo = "",                   # animated-sog checkout (default: next to this repo, else the installed package)
    [string]$EnvCmd = "C:\o4d\o4d-env.cmd",   # environment script from setup_env.ps1
    [switch]$SkipEncode,                      # kept for older command lines: only serve
    [switch]$Check,                           # also run `asog check` (error report against the .trigsbin)
    [switch]$Serve,                           # start a static server for the player (in this window)
    [int]$Port = 8765
)
$ErrorActionPreference = "Stop"

$Core = (Resolve-Path "$PSScriptRoot\..\..").Path
$Root = Split-Path $Core -Parent                                          # folder of both repos
if (-not $AsogRepo) {
    foreach ($d in "animated-sog-fork", "animated-sog") {
        if (Test-Path (Join-Path $Root "$d\python\src\asog")) { $AsogRepo = Join-Path $Root $d; break }
    }
}
$asogEnv = ""
if ($AsogRepo) {
    $AsogRepo = (Resolve-Path $AsogRepo).Path
    $asogEnv = "set `"PYTHONPATH=$AsogRepo\python\src`" && "
}
if ($Device -eq "cpu") { $asogEnv = "set `"CUDA_VISIBLE_DEVICES=-1`" && " + $asogEnv }

function Invoke-Asog([string[]]$AsogArgs) {
    $quoted = ($AsogArgs | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join " "
    # log on stdout (PowerShell 5.1 turns stderr lines into errors)
    cmd /d /c "call `"$EnvCmd`" >nul 2>nul && $asogEnv python -m asog.cli $quoted 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "asog $($AsogArgs[0]) failed ($LASTEXITCODE)" }
}

if ($Trigsbin -and -not $SkipEncode) {
    $Trigsbin = (Resolve-Path $Trigsbin).Path
    if (-not $Out) { $Out = [IO.Path]::ChangeExtension($Trigsbin, ".asog") }
    $Out = [IO.Path]::GetFullPath($Out)
    New-Item -ItemType Directory -Force (Split-Path $Out -Parent) | Out-Null
    $opts = @()
    if ($AsogRepo -and (Select-String -Path (Join-Path $AsogRepo "python\src\asog\cli.py") -Pattern "palette-device" -Quiet)) {
        $opts = @("--palette-device", $Device)
    }
    $t0 = Get-Date
    Invoke-Asog (@("encode", $Trigsbin, $Out) + $opts + $EncodeArgs)
    "encode: {0:n0} s" -f ((Get-Date) - $t0).TotalSeconds
    Invoke-Asog @("info", $Out)
    "{0}: {1:n1} MB  <-  {2}: {3:n1} MB" -f $Out, ((Get-Item $Out).Length / 1MB), $Trigsbin, ((Get-Item $Trigsbin).Length / 1MB)
    if ($Check) { Invoke-Asog @("check", $Trigsbin, $Out) }
}

# the player: serve the folder of both repos (player + data under one origin); three.js comes from the jsDelivr CDN
$player = if ($AsogRepo -and (Split-Path -Parent $AsogRepo) -eq $Root) { Split-Path -Leaf $AsogRepo } else { "animated-sog" }
if ($Out -and $Out.StartsWith($Root)) {
    $rel = $Out.Substring($Root.Length).TrimStart('\').Replace('\', '/')
    "player: http://localhost:$Port/$player/examples/three/?src=/$rel"
} else {
    "player: http://localhost:$Port/$player/examples/three/?src=/<path of the .asog under $Root>"
}
if ($Serve) {
    "serving $Root on port $Port (Ctrl+C stops)"
    cmd /d /c "call `"$EnvCmd`" >nul 2>nul && python -m http.server $Port --bind 127.0.0.1 --directory `"$Root`""
}
