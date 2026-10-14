<#
.SYNOPSIS
    Build the local CLRerNet environment in .clrernet/ (venv, checkout, patches).

.DESCRIPTION
    CLRerNet pins torch 2.2 and mmcv 2.2.0, which must not share an environment
    with the project's own or with ultralytics, so it gets its own venv. Nothing
    here is compiled: the one compiled piece, the CUDA lane NMS, is replaced by
    the NumPy port in scripts/clrernet/nms. Safe to re-run.

.EXAMPLE
    .\scripts\clrernet\setup.ps1
    .\.clrernet\venv\Scripts\python.exe scripts\clrernet\dump.py --route routes\kuching_01
#>

$ErrorActionPreference = 'Stop'
$root = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) '.clrernet'
$python = Join-Path $root 'venv\Scripts\python.exe'
$checkout = Join-Path $root 'CLRerNet'
# The last CLRerNet commit this was verified against; patch.py matches exact text.
$commit = 'dae038f67da57e292e5293a68c9c1c2922de13c2'

# torch is 2.3 GB and the OpenMMLab wheel server is slow; uv's 30 s default
# times out on both.
$env:UV_HTTP_TIMEOUT = '600'
$env:UV_LINK_MODE = 'copy'

function Invoke-Checked {
    param([string] $Exe, [string[]] $Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed (exit $LASTEXITCODE)" }
}

New-Item -ItemType Directory -Force $root | Out-Null
if (-not (Test-Path $python)) {
    Invoke-Checked uv @('venv', (Join-Path $root 'venv'), '--python', '3.11')
}

Invoke-Checked uv @('pip', 'install', '--python', $python,
    'torch==2.2.0', 'torchvision==0.17.0',
    '--index-url', 'https://download.pytorch.org/whl/cu121')
Invoke-Checked uv @('pip', 'install', '--python', $python,
    'numpy==1.26.4', 'opencv-python==4.9.0.80', 'mmengine==0.10.7', 'mmdet==3.3.0',
    'mmcv==2.2.0', 'yapf==0.40.1', 'setuptools==68.2.2',
    '--only-binary', 'mmcv',
    '-f', 'https://download.openmmlab.com/mmcv/dist/cu121/torch2.2.0/index.html')

if (-not (Test-Path $checkout)) {
    Invoke-Checked git @('clone', 'https://github.com/hirotomusiker/CLRerNet.git', $checkout)
    Invoke-Checked git @('-C', $checkout, 'checkout', '--quiet', $commit)
}

Invoke-Checked $python @((Join-Path $PSScriptRoot 'patch.py'), $checkout)
# No double quotes inside: Windows PowerShell 5.1 strips them from native arguments.
Invoke-Checked $python @('-c', 'import torch, mmcv, mmdet; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__, mmdet.__version__)')
