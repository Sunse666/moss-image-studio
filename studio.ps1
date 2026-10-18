param([int]$Port = 7860)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe studio\server.py $Port
