$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path .venv\Scripts\python.exe)) {
    py -3.13 -m venv .venv
    if ($LASTEXITCODE) { throw 'Install Python 3.13, then run setup.ps1 again.' }
}
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE) { throw 'pip setup failed' }
& $python -m pip install requests
if ($LASTEXITCODE) { throw "Download dependency installation failed" }
& $python install_cuda.py
if ($LASTEXITCODE) { throw 'CUDA PyTorch installation failed' }
& $python -m pip install --no-deps "git+https://github.com/Jit-Roy/WeeLLM.git@fbad760cfd2e3e4d1af213044f12b13c4414dfe6"
if ($LASTEXITCODE) { throw "WeeLLM installation failed" }
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE) { throw 'Dependency installation failed' }
& $python -c "import torch; from weellm import WeeTextToImagePipeline; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE) { throw 'CUDA / WeeLLM validation failed' }
Write-Host 'Setup complete. Run ./download.ps1, then ./studio.ps1'
