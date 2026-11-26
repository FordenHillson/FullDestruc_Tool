param(
    [string]$Source = "D:\backup\github\FullDestruc_Tool\addon",
    [string]$Destination = "C:\Users\saetanpee\AppData\Roaming\Blender Foundation\Blender\5.2\scripts\addons\FullDestruc_Tool"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Source)) {
    Write-Error "Source folder not found: $Source"
    exit 1
}

if (-not (Test-Path $Destination)) {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
}

# Copy addon source to Blender 5.2 addons folder, excluding __pycache__.
# NOTE: no /MIR — the installed folder may hold extra files (bin/*.dll, locally
# developed modules) that are not part of this repo; mirroring would delete them.
robocopy $Source $Destination /E /XD "__pycache__" ".git" /XF "*.pyc" "*.pyo" /NFL /NDL /NJH /NJS | Out-Null

# Robocopy exit codes 0-7 indicate success
if ($LASTEXITCODE -gt 7) {
    Write-Error "Sync failed with robocopy exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[OK] Synced FullDestruc_Tool -> $Destination" -ForegroundColor Green
exit 0
