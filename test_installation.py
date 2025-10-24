"""
Test Script - Verify Installation and Basic Functionality
Run this after setup to ensure everything works
"""

import sys
import os

print("="*60)
print("IMAGE GENERATION SYSTEM - VERIFICATION TEST")
print("="*60)

# Test 1: Python Version
print("\n1. Checking Python version...")
if sys.version_info >= (3, 8):
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
else:
    print(f"   ✗ Python version too old: {sys.version_info}")
    sys.exit(1)

# Test 2: Import Core Libraries
print("\n2. Testing core imports...")
required_modules = {
    'torch': 'PyTorch',
    'PIL': 'Pillow',
    'numpy': 'NumPy',
    'yaml': 'PyYAML',
    'diffusers': 'Diffusers',
    'transformers': 'Transformers',
    'gradio': 'Gradio'
}

failed_imports = []
for module, name in required_modules.items():
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError as e:
        print(f"   ✗ {name} - {e}")
        failed_imports.append(name)

if failed_imports:
    print(f"\n   ERROR: Missing packages: {', '.join(failed_imports)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: PyTorch and CUDA
print("\n3. Checking PyTorch and CUDA...")
import torch

print(f"   PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   ✓ VRAM: {vram:.2f} GB")
else:
    print("   ⚠ CUDA not available - will use CPU (slower)")

# Test 4: Configuration File
print("\n4. Checking configuration...")
try:
    from utils import load_config
    config = load_config()
    print("   ✓ config.yaml loaded")
    print(f"   Device: {config['hardware']['device']}")
    print(f"   Default steps: {config['generation']['default_steps']}")
except Exception as e:
    print(f"   ✗ Configuration error: {e}")
    sys.exit(1)

# Test 5: Directory Structure
print("\n5. Checking directories...")
required_dirs = [
    'data/prompts',
    'data/generated_images'
]

for directory in required_dirs:
    if os.path.exists(directory):
        print(f"   ✓ {directory}")
    else:
        print(f"   ⚠ Creating {directory}...")
        os.makedirs(directory, exist_ok=True)

# Test 6: Import Project Modules
print("\n6. Testing project modules...")
project_modules = [
    ('utils', 'Utilities'),
    ('model_loader', 'Model Loader'),
    ('image_generator', 'Image Generator'),
    ('evaluation', 'Evaluator')
]

for module, name in project_modules:
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except Exception as e:
        print(f"   ✗ {name} - {e}")

# Test 7: Model Loader Initialization
print("\n7. Testing model loader...")
try:
    from model_loader import ModelLoader
    loader = ModelLoader()
    print("   ✓ Model loader initialized")
except Exception as e:
    print(f"   ✗ Model loader error: {e}")

# Test 8: Memory Check
print("\n8. Checking available memory...")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    available = total - allocated
    print(f"   Total VRAM: {total:.2f} GB")
    print(f"   Available: {available:.2f} GB")
    
    # Recommendations
    if available >= 8:
        print("   ✓ Can run SDXL")
    elif available >= 4:
        print("   ✓ Can run Stable Diffusion and DreamShaper")
    else:
        print("   ⚠ Limited VRAM - recommend CPU offload")
else:
    print("   CPU mode - no VRAM limit")

# Test 9: Quick Generation Test (Optional)
print("\n9. Quick generation test (optional)")
test_generation = input("   Run a quick test generation? This will download a model (~4GB). (y/n): ").strip().lower()

if test_generation == 'y':
    print("\n   Testing image generation...")
    print("   Note: First run will download model (~4GB, one-time only)")
    print("   This may take 5-10 minutes on first run...")
    
    try:
        from image_generator import ImageGenerator
        
        generator = ImageGenerator()
        print("\n   Generating test image...")
        
        images = generator.generate(
            prompt="a simple test image",
            model="stable_diffusion",
            num_inference_steps=10,  # Quick test
            width=256,
            height=256
        )
        
        if images:
            print("   ✓ Test generation successful!")
            print(f"   Image saved to: data/generated_images/")
        else:
            print("   ✗ Test generation failed")
    
    except Exception as e:
        print(f"   ✗ Generation test failed: {e}")
        print("   This is normal on first run - try running main.py")
else:
    print("   Skipped test generation")

# Final Summary
print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)

if torch.cuda.is_available():
    print("\n✓ System ready for GPU-accelerated generation!")
    print("  Recommended: Start with SDXL for best quality")
else:
    print("\n✓ System ready for CPU generation!")
    print("  Recommended: Use Stable Diffusion for faster generation")

print("\nNext steps:")
print("  1. Run: python gradio_ui.py  (for web interface)")
print("  2. Or:  python main.py       (for interactive mode)")
print("  3. Read QUICKSTART.md for examples")

print("\nHappy generating! 🎨")
print("="*60 + "\n")