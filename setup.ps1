# Image Generation System - Setup Script for Windows
# Run this in PowerShell

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Image Generation System - Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Found: $pythonVersion" -ForegroundColor Green

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Check if virtual environment is activated
Write-Host "`nChecking virtual environment..." -ForegroundColor Yellow
if ($env:VIRTUAL_ENV) {
    Write-Host "Virtual environment active: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "WARNING: No virtual environment detected!" -ForegroundColor Yellow
    Write-Host "It's recommended to use a virtual environment." -ForegroundColor Yellow
    Write-Host "Run: .\LLM\Scripts\Activate.ps1" -ForegroundColor Yellow
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 0
    }
}

# Create directories
Write-Host "`nCreating project directories..." -ForegroundColor Yellow
$directories = @(
    "data/prompts",
    "data/generated_images",
    "models"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  Exists: $dir" -ForegroundColor Gray
    }
}

# Check for GPU
Write-Host "`nChecking for NVIDIA GPU..." -ForegroundColor Yellow
try {
    $gpu = nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  GPU detected: $gpu" -ForegroundColor Green
        $useGPU = $true
    } else {
        throw "No GPU"
    }
} catch {
    Write-Host "  No NVIDIA GPU detected - will use CPU mode" -ForegroundColor Yellow
    $useGPU = $false
}

# Install PyTorch
Write-Host "`nInstalling PyTorch..." -ForegroundColor Yellow
if ($useGPU) {
    Write-Host "  Installing GPU version (CUDA 11.8)..." -ForegroundColor Cyan
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
} else {
    Write-Host "  Installing CPU version..." -ForegroundColor Cyan
    pip install torch torchvision --quiet
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "  PyTorch installed successfully!" -ForegroundColor Green
} else {
    Write-Host "  ERROR: PyTorch installation failed!" -ForegroundColor Red
    exit 1
}

# Install other requirements
Write-Host "`nInstalling other dependencies..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host "  All dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "  WARNING: Some packages may have failed to install" -ForegroundColor Yellow
}

# Verify installation
Write-Host "`nVerifying installation..." -ForegroundColor Yellow

$imports = @(
    "torch",
    "diffusers",
    "transformers",
    "gradio",
    "PIL"
)

foreach ($module in $imports) {
    python -c "import $module" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $module" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $module" -ForegroundColor Red
    }
}

# Check CUDA availability
Write-Host "`nChecking CUDA..." -ForegroundColor Yellow
$cudaCheck = python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>&1
Write-Host "  $cudaCheck" -ForegroundColor $(if ($cudaCheck -match "True") { "Green" } else { "Yellow" })

# Create sample prompts file
Write-Host "`nCreating sample prompts..." -ForegroundColor Yellow
$samplePrompts = @"
A serene mountain landscape at sunset, digital art, highly detailed
A futuristic city with flying cars, cyberpunk style, neon lights
A cute robot playing with a cat in a garden, 3D render
An enchanted forest with glowing mushrooms, fantasy art
A steampunk airship in the clouds, Victorian era, detailed
"@

$samplePrompts | Out-File -FilePath "data/prompts/sample_prompts.txt" -Encoding UTF8
Write-Host "  Created: data/prompts/sample_prompts.txt" -ForegroundColor Green

# Final message
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nYou can now run the system using:" -ForegroundColor Green
Write-Host "  1. Interactive Mode:  python main.py" -ForegroundColor White
Write-Host "  2. Web UI:           python gradio_ui.py" -ForegroundColor White
Write-Host "  3. Generate Image:   python main.py generate 'your prompt'" -ForegroundColor White
Write-Host "`nFor help:            python main.py --help" -ForegroundColor White
Write-Host "`nCheck README.md for detailed documentation." -ForegroundColor Yellow
Write-Host ""