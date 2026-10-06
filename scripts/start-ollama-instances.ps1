<#
.SYNOPSIS
    Starts two Ollama servers, each pinned to one GPU.
      Instance A (primary): RTX 5070        -> 127.0.0.1:11434
      Instance B (memory) : RTX 2070 SUPER  -> 127.0.0.1:11435

.DESCRIPTION
    GPUs are pinned by UUID (CUDA device indices do not reliably follow slot
    order). By default the UUIDs are auto-detected from nvidia-smi by name.

    The Ollama tray app starts its own unpinned server on 11434. Quit it and
    disable its autostart first, or pass -Force to stop running Ollama processes.

.EXAMPLE
    .\start-ollama-instances.ps1 -ListGpus
    .\start-ollama-instances.ps1 -Force
    .\start-ollama-instances.ps1 -Status
    .\start-ollama-instances.ps1 -Stop
#>
[CmdletBinding()]
param(
    [string]$PrimaryGpu = "",              # UUID (GPU-xxxx...). Empty = auto-detect by -PrimaryMatch
    [string]$MemoryGpu  = "",
    [string]$PrimaryMatch = "5070",
    [string]$MemoryMatch  = "2070",
    [int]$PrimaryPort = 11434,
    [int]$MemoryPort  = 11435,
    [int]$PrimaryContext = 16384,
    [int]$MemoryContext  = 8192,
    [string]$ModelsDir = $env:OLLAMA_MODELS,   # e.g. G:\AI\models - shared by both instances
    [string]$LogDir = "G:\AI\logs",
    [string]$KvCacheType = "q8_0",             # halves KV-cache VRAM vs f16; needs flash attention
    [switch]$ListGpus,
    [switch]$Status,
    [switch]$Stop,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$pidFile = Join-Path $LogDir "ollama-instances.json"

function Get-Gpus {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { throw "nvidia-smi not found. Install/repair the NVIDIA driver." }
    & nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader | ForEach-Object {
        $p = $_.Split(",") | ForEach-Object { $_.Trim() }
        [pscustomobject]@{ Index = $p[0]; Name = $p[1]; Uuid = $p[2]; Memory = $p[3] }
    }
}

function Resolve-Gpu([string]$uuid, [string]$match, $gpus, [string]$role) {
    if ($uuid) { return $uuid }
    $hits = @($gpus | Where-Object { $_.Name -match $match })
    if ($hits.Count -ne 1) {
        throw "Could not uniquely find the $role GPU matching '$match'. Pass -$($role)Gpu <UUID> (see -ListGpus)."
    }
    return $hits[0].Uuid
}

function Test-Port([int]$port) {
    [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

function Wait-Ollama([int]$port) {
    for ($i = 0; $i -lt 40; $i++) {
        try { return (Invoke-RestMethod "http://127.0.0.1:$port/api/version" -TimeoutSec 2).version }
        catch { Start-Sleep -Milliseconds 500 }
    }
    return $null
}

if ($ListGpus) { Get-Gpus | Format-Table -AutoSize; return }

if ($Status) {
    foreach ($port in @($PrimaryPort, $MemoryPort)) {
        Write-Host "`n== 127.0.0.1:$port ==" -ForegroundColor Cyan
        try {
            $ps = Invoke-RestMethod "http://127.0.0.1:$port/api/ps" -TimeoutSec 3
            if (-not $ps.models) { Write-Host "  (no model loaded)" }
            foreach ($m in $ps.models) {
                $pct = if ($m.size) { [math]::Round(100 * $m.size_vram / $m.size, 1) } else { 0 }
                $color = if ($pct -ge 100) { "Green" } else { "Yellow" }
                Write-Host ("  {0,-24} {1,6:N1} GB  {2,5}% GPU  ctx {3}" -f $m.name, ($m.size / 1GB), $pct, $m.context_length) -ForegroundColor $color
                if ($pct -lt 100) { Write-Host "  WARNING: partly on CPU. Lower num_ctx or use a smaller quant." -ForegroundColor Yellow }
            }
        } catch { Write-Host "  not reachable" -ForegroundColor Red }
    }
    Write-Host ""
    & nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    return
}

if ($Stop) {
    if (Test-Path $pidFile) {
        (Get-Content $pidFile | ConvertFrom-Json) | ForEach-Object {
            Stop-Process -Id $_.Pid -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped $($_.Role) (PID $($_.Pid))"
        }
        Remove-Item $pidFile -Force
    } else { Write-Host "No PID file at $pidFile" }
    return
}

$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) { throw "ollama.exe not on PATH." }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$gpus = Get-Gpus
$pGpu = Resolve-Gpu $PrimaryGpu $PrimaryMatch $gpus "Primary"
$mGpu = Resolve-Gpu $MemoryGpu  $MemoryMatch  $gpus "Memory"
if ($pGpu -eq $mGpu) { throw "Primary and memory resolved to the same GPU ($pGpu)." }

foreach ($port in @($PrimaryPort, $MemoryPort)) {
    if (Test-Port $port) {
        if ($Force) {
            Write-Host "Port $port busy; stopping existing Ollama processes (-Force)..." -ForegroundColor Yellow
            Get-Process "ollama app", "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
            Start-Sleep 2
            break
        }
        throw "Port $port is already in use (probably the Ollama tray app). Quit it, disable its autostart, or rerun with -Force."
    }
}

$instances = @(
    @{ Role = "primary"; Port = $PrimaryPort; Gpu = $pGpu; Ctx = $PrimaryContext },
    @{ Role = "memory";  Port = $MemoryPort;  Gpu = $mGpu; Ctx = $MemoryContext }
)
$vars = "OLLAMA_HOST","CUDA_VISIBLE_DEVICES","OLLAMA_MODELS","OLLAMA_FLASH_ATTENTION",
        "OLLAMA_KV_CACHE_TYPE","OLLAMA_NUM_PARALLEL","OLLAMA_MAX_LOADED_MODELS",
        "OLLAMA_CONTEXT_LENGTH","OLLAMA_KEEP_ALIVE"
$saved = @{}; foreach ($v in $vars) { $saved[$v] = [Environment]::GetEnvironmentVariable($v, "Process") }

$started = @()
try {
    foreach ($i in $instances) {
        # Child processes inherit this session's environment at Start-Process time.
        $env:OLLAMA_HOST              = "127.0.0.1:$($i.Port)"
        $env:CUDA_VISIBLE_DEVICES     = $i.Gpu
        $env:OLLAMA_FLASH_ATTENTION   = "1"
        $env:OLLAMA_KV_CACHE_TYPE     = $KvCacheType
        $env:OLLAMA_NUM_PARALLEL      = "1"   # each parallel slot costs a full KV cache
        $env:OLLAMA_MAX_LOADED_MODELS = "1"
        $env:OLLAMA_CONTEXT_LENGTH    = "$($i.Ctx)"
        $env:OLLAMA_KEEP_ALIVE        = "30m"
        if ($ModelsDir) { $env:OLLAMA_MODELS = $ModelsDir }

        $p = Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden -PassThru `
              -RedirectStandardError  (Join-Path $LogDir "ollama-$($i.Role).log") `
              -RedirectStandardOutput (Join-Path $LogDir "ollama-$($i.Role).out.log")
        $started += [pscustomobject]@{ Role = $i.Role; Pid = $p.Id; Port = $i.Port; Gpu = $i.Gpu }
    }
} finally {
    foreach ($v in $vars) { [Environment]::SetEnvironmentVariable($v, $saved[$v], "Process") }
}
$started | ConvertTo-Json | Set-Content $pidFile

foreach ($s in $started) {
    $ver = Wait-Ollama $s.Port
    $name = ($gpus | Where-Object Uuid -eq $s.Gpu).Name
    if ($ver) { Write-Host ("{0,-8} 127.0.0.1:{1}  Ollama {2}  -> {3}" -f $s.Role, $s.Port, $ver, $name) -ForegroundColor Green }
    else { Write-Host "$($s.Role) did not come up; see $LogDir\ollama-$($s.Role).log" -ForegroundColor Red }
}
Write-Host "`nLoad a model on each, then verify pinning with:  .\start-ollama-instances.ps1 -Status"
