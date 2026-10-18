param([ValidateSet('Default','SFT','Base','Turbo','INT8')][string]$Model = 'Default')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe download.py $Model
if ($LASTEXITCODE) { throw 'Model download failed; rerun to resume.' }
