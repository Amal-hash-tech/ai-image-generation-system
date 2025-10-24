"""
Model Loader for Image Generation System
Handles loading Stable Diffusion, SDXL, ControlNet, and DreamShaper models
"""

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)
from utils import load_config, setup_device
import warnings
warnings.filterwarnings('ignore')


class ModelLoader:
    """Load and manage different image generation models"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize model loader with configuration"""
        self.config = load_config(config_path)
        self.device, self.dtype = setup_device(self.config)
        self.loaded_models = {}
        
        print("\n" + "="*60)
        print("IMAGE GENERATION SYSTEM - MODEL LOADER")
        print("="*60)
    
    def load_stable_diffusion(self):
        """Load Stable Diffusion 1.5 model"""
        if 'stable_diffusion' in self.loaded_models:
            print("✓ Stable Diffusion already loaded")
            return self.loaded_models['stable_diffusion']
        
        print("\n📥 Loading Stable Diffusion v1.5...")
        model_id = self.config['models']['stable_diffusion']['model_id']
        
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None,  # Disable for speed
                requires_safety_checker=False
            )
            
            # Optimize
            pipe = pipe.to(self.device)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            
            # Enable memory optimizations
            if self.config['hardware'].get('enable_xformers', False) and self.device.type == 'cuda':
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    print("  ✓ XFormers enabled")
                except:
                    print("  ⚠ XFormers not available")
            
            if self.config['hardware'].get('enable_cpu_offload', False):
                pipe.enable_model_cpu_offload()
                print("  ✓ CPU offload enabled")
            
            self.loaded_models['stable_diffusion'] = pipe
            print("✓ Stable Diffusion v1.5 loaded successfully!")
            return pipe
            
        except Exception as e:
            print(f"✗ Error loading Stable Diffusion: {e}")
            return None
    
    def load_sdxl(self):
        """Load Stable Diffusion XL model"""
        if 'sdxl' in self.loaded_models:
            print("✓ SDXL already loaded")
            return self.loaded_models['sdxl']
        
        print("\n📥 Loading Stable Diffusion XL...")
        model_id = self.config['models']['sdxl']['model_id']
        
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16" if self.dtype == torch.float16 else None
            )
            
            pipe = pipe.to(self.device)
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            
            # Enable optimizations
            if self.config['hardware'].get('enable_xformers', False) and self.device.type == 'cuda':
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    print("  ✓ XFormers enabled")
                except:
                    print("  ⚠ XFormers not available")
            
            self.loaded_models['sdxl'] = pipe
            print("✓ SDXL loaded successfully!")
            return pipe
            
        except Exception as e:
            print(f"✗ Error loading SDXL: {e}")
            return None
    
    def load_controlnet(self, controlnet_type="canny"):
        """Load ControlNet model"""
        key = f'controlnet_{controlnet_type}'
        if key in self.loaded_models:
            print(f"✓ ControlNet ({controlnet_type}) already loaded")
            return self.loaded_models[key]
        
        print(f"\n📥 Loading ControlNet ({controlnet_type})...")
        
        try:
            # Load ControlNet
            controlnet_id = self.config['models']['controlnet']['model_id']
            controlnet = ControlNetModel.from_pretrained(
                controlnet_id,
                torch_dtype=self.dtype
            )
            
            # Load base Stable Diffusion with ControlNet
            base_model_id = self.config['models']['stable_diffusion']['model_id']
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                base_model_id,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                safety_checker=None
            )
            
            pipe = pipe.to(self.device)
            
            self.loaded_models[key] = pipe
            print(f"✓ ControlNet ({controlnet_type}) loaded successfully!")
            return pipe
            
        except Exception as e:
            print(f"✗ Error loading ControlNet: {e}")
            return None
    
    def load_dreamshaper(self):
        """Load DreamShaper model"""
        if 'dreamshaper' in self.loaded_models:
            print("✓ DreamShaper already loaded")
            return self.loaded_models['dreamshaper']
        
        print("\n📥 Loading DreamShaper...")
        model_id = self.config['models']['dreamshaper']['model_id']
        
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None
            )
            
            pipe = pipe.to(self.device)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            
            self.loaded_models['dreamshaper'] = pipe
            print("✓ DreamShaper loaded successfully!")
            return pipe
            
        except Exception as e:
            print(f"✗ Error loading DreamShaper: {e}")
            return None
    
    def load_model(self, model_name):
        """Load a specific model by name"""
        model_loaders = {
            'stable_diffusion': self.load_stable_diffusion,
            'sd': self.load_stable_diffusion,
            'sdxl': self.load_sdxl,
            'controlnet': self.load_controlnet,
            'dreamshaper': self.load_dreamshaper
        }
        
        loader = model_loaders.get(model_name.lower())
        if loader:
            return loader()
        else:
            print(f"✗ Unknown model: {model_name}")
            print(f"Available models: {', '.join(model_loaders.keys())}")
            return None
    
    def unload_model(self, model_name):
        """Unload a model to free memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"✓ {model_name} unloaded")
        else:
            print(f"⚠ {model_name} not loaded")
    
    def list_loaded_models(self):
        """List currently loaded models"""
        if not self.loaded_models:
            print("No models currently loaded")
        else:
            print("\nLoaded models:")
            for model_name in self.loaded_models.keys():
                print(f"  • {model_name}")
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\nGPU Memory:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            return allocated, reserved
        else:
            print("CPU mode - memory tracking not available")
            return None, None


# Example usage
if __name__ == "__main__":
    loader = ModelLoader()
    
    # Load a model
    print("\nTesting model loader...")
    pipe = loader.load_stable_diffusion()
    
    if pipe:
        print("\n✓ Model loader test successful!")
        loader.list_loaded_models()
        loader.get_memory_usage()