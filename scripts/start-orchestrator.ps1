<#
.SYNOPSIS
    Creates the virtual environment on first run, then starts the orchestrator
    on http://127.0.0.1:8000 (Ollama-compatible for OpenClaw + the /chat API).
.EXAMPLE
    .\start-orchestrator.ps1            # install if needed, then serve
    .\start-orchestrator.ps1 -Install   # install/refresh dependencies only
    .\start-orchestrator.ps1 -Test      # run the test suite
#>
param([switch]$Install, [switch]$Test)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    # Works in Windows PowerShell 5.1 and PowerShell 7.
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { & $launcher.Source -3 -m venv .venv }
    else {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) { throw "Python 3.11+ not found. Install it from python.org (tick 'Add to PATH')." }
        & $python.Source -m venv .venv
    }
    $Install = $true
}
if ($Install -or $Test) {
    & $venvPy -m pip install --upgrade pip -q
    & $venvPy -m pip install -r requirements-dev.txt -q
}
if ($Test) { & $venvPy -m pytest -q; exit $LASTEXITCODE }
if ($Install -and $PSBoundParameters.ContainsKey("Install")) { Write-Host "Dependencies installed."; exit 0 }

& $venvPy -m app.main
