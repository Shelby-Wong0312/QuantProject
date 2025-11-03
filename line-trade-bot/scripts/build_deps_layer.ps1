Param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$build = Join-Path $root "terraform\build"
$layerRoot = Join-Path $build "deps"
$layerPy = Join-Path $layerRoot "python"
$zipPath = Join-Path $build "deps_layer.zip"

New-Item -ItemType Directory -Force -Path $layerPy | Out-Null

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install "line-bot-sdk>=2,<3" -t $layerPy

if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path (Join-Path $layerRoot "python") -DestinationPath $zipPath -Force
Write-Host "Built deps layer: $zipPath"

