param([ValidateSet('SFT','Base','Turbo')][string]$Model = 'SFT')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe download.py $Model
if ($LASTEXITCODE) { throw 'Model download failed; rerun to resume.' }
