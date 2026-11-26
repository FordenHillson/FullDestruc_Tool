param(
    [string]$Installed = "C:\Users\saetanpee\AppData\Roaming\Blender Foundation\Blender\5.2\scripts\addons\FullDestruc_Tool",
    [switch]$SkipSync
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoAddon = Join-Path $repoRoot "addon"
$dllName = "fdt_bool_engine.dll"
$srcDll = Join-Path $repoAddon "bin\$dllName"
$dstDir = Join-Path $Installed "bin"
$dstDll = Join-Path $dstDir $dllName

if (-not (Test-Path -LiteralPath $srcDll)) {
    Write-Error "Source engine DLL not found: $srcDll"
    exit 1
}

# 1) Sync all addon Python files first (keeps bridge/UI/core in step with the DLL).
if (-not $SkipSync) {
    $sync = Join-Path $PSScriptRoot "sync_to_blender.ps1"
    & $sync -Source $repoAddon -Destination $Installed
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# 2) Restore the compiled Rust engine next to the installed addon modules.
if (-not (Test-Path -LiteralPath $dstDir)) {
    New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
}
Copy-Item -LiteralPath $srcDll -Destination $dstDll -Force

$srcHash = (Get-FileHash -LiteralPath $srcDll -Algorithm SHA256).Hash
$dstHash = (Get-FileHash -LiteralPath $dstDll -Algorithm SHA256).Hash
if ($srcHash -ne $dstHash) {
    Write-Error "DLL copy verification failed. Source $srcHash != installed $dstHash"
    exit 1
}
Write-Host "[OK] Restored $dllName -> $dstDll (SHA256 $dstHash)"

# 3) Verify the installed DLL actually loads and exposes the required ABI.
$python = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $python) {
    $ver = & python -c "import ctypes,sys; print(ctypes.CDLL(sys.argv[1]).fdt_engine_version())" $dstDll
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Installed DLL failed to load in Python"
        exit 1
    }
    $verInt = [int]$ver
    if ($verInt -lt 140) {
        Write-Error "Incompatible engine DLL version $verInt (need >= 140)"
        exit 1
    }

    $missing = & python -c "import ctypes,sys; d=ctypes.CDLL(sys.argv[1]); print(','.join(n for n in ['fdt_engine_version','fdt_session_create','fdt_session_destroy','fdt_session_set_target','fdt_session_eval_cutters','fdt_eval_free','fdt_session_submit_async','fdt_session_poll_gpu_preview','fdt_gpu_preview_free','fdt_session_fetch_cached_eval'] if not hasattr(d,n)))" $dstDll
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to inspect DLL exports"
        exit 1
    }
    if (-not [string]::IsNullOrWhiteSpace($missing)) {
        Write-Error "Installed DLL is missing exports: $missing"
        exit 1
    }
    Write-Host "[OK] Installed DLL loads: version $verInt, all required exports present"
} else {
    Write-Warning "python not found on PATH - skipped load/export verification"
}

Write-Host "[OK] Rust engine restore complete. Restart Blender, then pick 'Rust Engine' in N-Panel > FullDestruc or the Python Window."
exit 0
