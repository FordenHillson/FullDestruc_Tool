$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BinDir = Join-Path (Split-Path -Parent $ScriptDir) "bin"

Write-Host "[FDT Rust Engine v2.1] Building high-scale release DLL (4M-Tri Chunk Ownership + Async Target)..." -ForegroundColor Cyan
Push-Location $ScriptDir
try {
    cargo build --release
    if (!(Test-Path $BinDir)) {
        New-Item -ItemType Directory -Path $BinDir | Out-Null
    }
    $SrcDll = Join-Path $ScriptDir "target\release\fdt_bool_engine.dll"
    $DstDll = Join-Path $BinDir "fdt_bool_engine.dll"

    # Clean up any old unlocked hot-swap backups
    Get-ChildItem $BinDir -Filter "fdt_bool_engine.dll.old_*" -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
    }

    try {
        Copy-Item -Path $SrcDll -Destination $DstDll -Force -ErrorAction Stop
    } catch {
        # If Blender currently holds the DLL loaded, hot-swap by renaming the old DLL first
        $OldDll = "$DstDll.old_$(Get-Date -Format 'HHmmss')"
        Move-Item -Path $DstDll -Destination $OldDll -Force
        Copy-Item -Path $SrcDll -Destination $DstDll -Force
        Write-Host "[FDT Rust Engine] Hot-swapped locked DLL (old moved to $OldDll)" -ForegroundColor Yellow
    }
    Write-Host "[FDT Rust Engine] Deployed to: $DstDll" -ForegroundColor Green
} finally {
    Pop-Location
}
