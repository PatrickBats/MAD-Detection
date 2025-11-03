# Activate virtual environment for MAD-Detection project (PowerShell)
.\venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  MAD-Detection Virtual Environment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "PyTorch with CUDA 12.6 installed"
Write-Host "GPU: RTX 4070"
Write-Host ""
Write-Host "Quick commands:" -ForegroundColor Yellow
Write-Host "  - python estimate_training_time.py   (estimate training time)"
Write-Host "  - cd FullSynthetic\DDPM-MNIST        (go to FullSynthetic)"
Write-Host "  - python main.py                     (run FullSynthetic training)"
Write-Host ""
Write-Host "  - cd Fresh\MNIST-DDPM                (go to Fresh)"
Write-Host "  - python newmain.py                  (run Fresh training)"
Write-Host ""
Write-Host "To deactivate: type 'deactivate'" -ForegroundColor Yellow
Write-Host ""
