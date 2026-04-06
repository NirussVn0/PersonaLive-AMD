<#
.SYNOPSIS
    PersonaLive HIP/ZLUDA one-click launcher for AMD GPUs on Windows.
.DESCRIPTION
    Validates AMD HIP SDK installation, configures environment variables
    for RDNA2 (gfx1030), activates conda environment, launches the
    inference server, and opens the browser automatically.
#>

param(
    [string]$CondaEnv = "personalive",
    [int]$DefaultPort = 7860,
    [string]$Host = "0.0.0.0",
    [string]$ConfigPath = "./configs/prompts/personalive_online.yaml"
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

Set-Location $ScriptRoot

function Write-Status {
    param([string]$Message, [string]$Color = "Cyan")
    Write-Host "[PersonaLive] " -ForegroundColor DarkGray -NoNewline
    Write-Host $Message -ForegroundColor $Color
}

function Write-Failure {
    param([string]$Message)
    Write-Host "[PersonaLive] " -ForegroundColor DarkGray -NoNewline
    Write-Host "ERROR: $Message" -ForegroundColor Red
}

function Test-HipSdk {
    try {
        $hipconfig = Get-Command hipconfig -ErrorAction SilentlyContinue
        if ($null -eq $hipconfig) {
            return $false
        }
        $version = & hipconfig --version 2>$null
        Write-Status "AMD HIP SDK detected: $version" "Green"
        return $true
    }
    catch {
        return $false
    }
}

function Test-ZludaDlls {
    $pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($null -eq $pythonPath) {
        return $false
    }

    $pythonDir = Split-Path -Parent $pythonPath
    $zludaPaths = @(
        (Join-Path $pythonDir "nvcuda.dll"),
        (Join-Path $pythonDir "Lib" "site-packages" "zluda" "nvcuda.dll")
    )

    foreach ($path in $zludaPaths) {
        if (Test-Path $path) {
            Write-Status "ZLUDA DLL found: $path" "Green"
            return $true
        }
    }

    $envPaths = $env:PATH -split ";"
    foreach ($dir in $envPaths) {
        $candidate = Join-Path $dir "nvcuda.dll"
        if (Test-Path $candidate) {
            Write-Status "ZLUDA DLL found in PATH: $candidate" "Green"
            return $true
        }
    }

    return $false
}

function Find-AvailablePort {
    param([int]$StartPort = 7860, [int]$MaxAttempts = 10)

    for ($i = 0; $i -lt $MaxAttempts; $i++) {
        $port = $StartPort + $i
        $listener = $null
        try {
            $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
            $listener.Start()
            $listener.Stop()
            return $port
        }
        catch {
            if ($null -ne $listener) {
                try { $listener.Stop() } catch {}
            }
            continue
        }
    }
    return $StartPort
}

function Set-HipEnvironment {
    $env:HIP_VISIBLE_DEVICES = "0"
    $env:HSA_OVERRIDE_GFX_VERSION = "10.3.0"
    $env:PYTORCH_HIP_ALLOC_CONF = "garbage_collection_threshold:0.9,max_split_size_mb:512"
    $env:HIP_FORCE_DEV_KERNARG = "1"
    $env:PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.9,max_split_size_mb:512"

    Write-Status "HIP environment configured:" "Yellow"
    Write-Status "  HIP_VISIBLE_DEVICES      = $env:HIP_VISIBLE_DEVICES"
    Write-Status "  HSA_OVERRIDE_GFX_VERSION = $env:HSA_OVERRIDE_GFX_VERSION"
    Write-Status "  PYTORCH_HIP_ALLOC_CONF   = $env:PYTORCH_HIP_ALLOC_CONF"
    Write-Status "  HIP_FORCE_DEV_KERNARG    = $env:HIP_FORCE_DEV_KERNARG"
}

function Activate-CondaEnvironment {
    param([string]$EnvName)

    $condaExe = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -eq $condaExe) {
        $condaPaths = @(
            "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
            "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
            "C:\ProgramData\miniconda3\Scripts\conda.exe",
            "C:\ProgramData\anaconda3\Scripts\conda.exe"
        )
        foreach ($path in $condaPaths) {
            if (Test-Path $path) {
                $condaExe = $path
                break
            }
        }
    }

    if ($null -ne $condaExe) {
        $condaRoot = Split-Path -Parent (Split-Path -Parent $condaExe)
        $hookScript = Join-Path $condaRoot "shell" "condabin" "conda-hook.ps1"
        if (Test-Path $hookScript) {
            & $hookScript
        }
        conda activate $EnvName 2>$null
        Write-Status "Conda environment '$EnvName' activated" "Green"
        return
    }

    $venvPath = Join-Path $ScriptRoot "venv" "Scripts" "Activate.ps1"
    if (Test-Path $venvPath) {
        & $venvPath
        Write-Status "Virtual environment activated" "Green"
        return
    }

    Write-Status "No conda or venv found, using system Python" "Yellow"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  PersonaLive - AMD HIP/ZLUDA Edition  " -ForegroundColor Magenta
Write-Host "  RX 6800 (gfx1030, RDNA2, 16GB)      " -ForegroundColor DarkMagenta
Write-Host "========================================" -ForegroundColor Magenta
Write-Host ""

Write-Status "Checking prerequisites..."

if (-not (Test-HipSdk)) {
    Write-Status "AMD HIP SDK not found (hipconfig not in PATH)" "Yellow"
    Write-Status "This may still work if ZLUDA is properly configured" "Yellow"
}

if (-not (Test-ZludaDlls)) {
    Write-Status "ZLUDA DLLs not detected in Python environment or PATH" "Yellow"
    Write-Status "Ensure ZLUDA is installed. See hip_setup_guide.md" "Yellow"
}

Activate-CondaEnvironment -EnvName $CondaEnv

Set-HipEnvironment

$port = Find-AvailablePort -StartPort $DefaultPort
if ($port -ne $DefaultPort) {
    Write-Status "Port $DefaultPort in use, using port $port instead" "Yellow"
}

Write-Status "Starting PersonaLive on port $port..." "Green"

$browserJob = Start-Job -ScriptBlock {
    param($port)
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:$port"
} -ArgumentList $port

try {
    python inference_online.py --port $port --host $Host --config_path $ConfigPath --acceleration none
}
catch {
    Write-Failure "Server process terminated: $_"
}
finally {
    Write-Status "Shutting down..." "Yellow"

    if ($null -ne $browserJob) {
        Stop-Job -Job $browserJob -ErrorAction SilentlyContinue
        Remove-Job -Job $browserJob -ErrorAction SilentlyContinue
    }

    $pythonProcs = Get-Process -Name python -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -eq "" }
    if ($null -ne $pythonProcs) {
        Write-Status "Cleaning up orphaned Python processes..."
        $pythonProcs | Stop-Process -Force -ErrorAction SilentlyContinue
    }

    Write-Status "Cleanup complete" "Green"
}
