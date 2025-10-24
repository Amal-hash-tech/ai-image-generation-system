"""
Utility functions for the Image Generation System
"""

import os
import yaml
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image
import json


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    """Setup computation device (GPU/CPU)"""
    device_str = config['hardware']['device']
    
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU (Note: Generation will be slower)")
    
    # Determine dtype
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    return device, dtype


def create_output_directory(base_path):
    """Create output directory structure"""
    Path(base_path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory created: {base_path}")
    return base_path


def save_image_with_metadata(image, prompt, output_path, metadata=None):
    """Save image with generation metadata"""
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_{timestamp}.png"
    filepath = os.path.join(output_path, filename)
    
    # Save image
    image.save(filepath)
    
    # Save metadata
    if metadata:
        metadata_file = filepath.replace('.png', '_metadata.json')
        metadata_dict = {
            'prompt': prompt,
            'timestamp': timestamp,
            **metadata
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    print(f"✓ Image saved: {filename}")
    return filepath


def display_image(image):
    """Display image (for notebooks/UI)"""
    if isinstance(image, str):
        image = Image.open(image)
    return image


def calculate_memory_requirements(model_type):
    """Estimate memory requirements for different models"""
    requirements = {
        'stable_diffusion': {'vram': 4, 'ram': 8},
        'sdxl': {'vram': 8, 'ram': 16},
        'controlnet': {'vram': 6, 'ram': 12},
        'dreamshaper': {'vram': 4, 'ram': 8}
    }
    return requirements.get(model_type, {'vram': 4, 'ram': 8})


def validate_prompt(prompt):
    """Validate and clean the input prompt"""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    # Clean prompt
    prompt = prompt.strip()
    
    # Warn if too long
    if len(prompt) > 500:
        print("⚠ Warning: Very long prompt. Consider shortening for better results.")
    
    return prompt


def get_available_models():
    """Get list of available/downloaded models"""
    models = []
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            if item.startswith("models--"):
                model_name = item.replace("models--", "").replace("--", "/")
                models.append(model_name)
    
    return models


def format_generation_params(params):
    """Format generation parameters for display"""
    formatted = []
    for key, value in params.items():
        formatted.append(f"{key}: {value}")
    return "\n".join(formatted)


class ProgressTracker:
    """Simple progress tracker for generation"""
    
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
    
    def update(self, step=None):
        """Update progress"""
        if step:
            self.current_step = step
        else:
            self.current_step += 1
        
        progress = (self.current_step / self.total_steps) * 100
        print(f"\rProgress: {progress:.1f}% ({self.current_step}/{self.total_steps})", end='')
        
        if self.current_step >= self.total_steps:
            print()  # New line when complete