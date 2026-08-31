<#
.SYNOPSIS
    Populate a route's detections/ directory: CLRerNet lanes and YOLO signs.

.DESCRIPTION
    This is the one step that needs models. Once it has written its two JSONL
    artifacts, every `irap report` and threshold sweep runs from that cache in
    seconds, with no GPU and no model install.

    Lane detections are NOT produced here. CLRerNet needs an mmcv-pinned
    environment this project does not reproduce, so inference happens in
    notebooks/clrernet_export.ipynb on Colab; this script converts the dump that
    notebook writes.

.PARAMETER Route
    The route directory: frames.csv, frames/, and (for -Report) gps.csv, imu.csv.

.PARAMETER Raw
    The clrernet_raw.json produced by the Colab notebook.
    Defaults to clrernet_raw.json inside the route directory.

.PARAMETER Weights
    The fine-tuned YOLO sign weight.

.PARAMETER SkipLanes
    Run signs only. Required to proceed without a dump: a route with no
    lane.jsonl scores every segment as the worst contrast band, so skipping
    lanes must be deliberate rather than the consequence of a missing file.

.PARAMETER SkipSigns
    Run the lane import only.

.PARAMETER Report
    Also run `irap report` afterwards. Needs gps.csv and imu.csv in the route.

.EXAMPLE
    .\scripts\run_detectors.ps1 -Route D:\routes\kuching_01 -Report

.EXAMPLE
    .\scripts\run_detectors.ps1 -Route D:\routes\kuching_01 -Raw D:\downloads\clrernet_raw.json
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Route,

    [string] $Raw,

    [string] $Weights = "G:\My Drive\FYP_runs\yolo11m_5class_v1\weights\best.pt",

    [string] $Out = "out",

    [switch] $SkipLanes,
    [switch] $SkipSigns,
    [switch] $Report
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Step {
    param([string] $Title, [string[]] $UvArgs)

    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    Write-Host "    uv $($UvArgs -join ' ')" -ForegroundColor DarkGray
    & uv @UvArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Title failed (exit $LASTEXITCODE)"
    }
}

# uv installs to the WinGet links directory, which a fresh shell does not always
# carry on PATH.
if ($null -eq (Get-Command uv -ErrorAction SilentlyContinue)) {
    $wingetLinks = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links'
    if (Test-Path (Join-Path $wingetLinks 'uv.exe')) {
        $env:PATH = "$env:PATH;$wingetLinks"
    }
    else {
        throw "uv not found on PATH and not at $wingetLinks"
    }
}

if (-not (Test-Path $Route)) { throw "Route directory not found: $Route" }
$Route = (Resolve-Path $Route).Path
if (-not (Test-Path (Join-Path $Route 'frames.csv'))) {
    throw "No frames.csv in $Route - the runners key every detection by its frame_id"
}

if (-not $Raw) { $Raw = Join-Path $Route 'clrernet_raw.json' }

# Refuse to fall through to a lane-less run by accident. A missing lane.jsonl is
# not an error downstream, it is a route with no markings anywhere, which grades
# as critical throughout.
if (-not $SkipLanes -and -not (Test-Path $Raw)) {
    Write-Host "CLRerNet dump not found: $Raw" -ForegroundColor Red
    Write-Host ""
    Write-Host "Produce one by running notebooks/clrernet_export.ipynb in Colab, or"
    Write-Host "pass -SkipLanes to run the sign detector alone. Without lane.jsonl"
    Write-Host "every segment reads as unmarked road and grades critical."
    exit 1
}
if (-not $SkipSigns -and -not (Test-Path $Weights)) {
    throw "Sign weight not found: $Weights"
}

Push-Location $projectRoot
try {
    Invoke-Step 'Syncing the runners extra (ultralytics)' @('sync', '--extra', 'dev', '--extra', 'runners')

    if (-not $SkipLanes) {
        Invoke-Step 'Importing CLRerNet lanes' @('run', 'irap', 'import-lanes', '--route', $Route, '--raw', $Raw)
    }
    if (-not $SkipSigns) {
        Invoke-Step 'Detecting traffic signs' @('run', 'irap', 'detect-signs', '--route', $Route, '--weights', $Weights)
    }

    Write-Host ""
    Write-Host "Detections in $Route\detections:" -ForegroundColor Green
    Get-ChildItem (Join-Path $Route 'detections') -Filter *.jsonl |
        ForEach-Object {
            $lines = (Get-Content $_.FullName | Measure-Object -Line).Lines
            Write-Host ("  {0,-16} {1,6} records" -f $_.Name, $lines)
        }

    if ($Report) {
        Invoke-Step 'Running the full CPU layer' @('run', 'irap', 'report', '--route', $Route, '--out', $Out)
    }
    else {
        Write-Host ""
        Write-Host "Next:  uv run irap report --route `"$Route`" --out $Out" -ForegroundColor DarkGray
    }
}
finally {
    Pop-Location
}
