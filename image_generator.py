"""
Image Generator - Main generation logic for all models
"""

import torch
from PIL import Image
import numpy as np
from model_loader import ModelLoader
from utils import (
    load_config, 
    validate_prompt, 
    save_image_with_metadata,
    create_output_directory
)
from datetime import datetime
import os


class ImageGenerator:
    """Main class for generating images from text prompts"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize the image generator"""
        self.config = load_config(config_path)
        self.model_loader = ModelLoader(config_path)
        self.output_path = create_output_directory(
            self.config['output']['save_path']
        )
        
        print("\n" + "="*60)
        print("IMAGE GENERATION SYSTEM READY")
        print("="*60 + "\n")
    
    def generate(
        self,
        prompt,
        model="stable_diffusion",
        negative_prompt="",
        num_inference_steps=None,
        guidance_scale=None,
        width=None,
        height=None,
        seed=None,
        num_images=1
    ):
        """
        Generate image(s) from text prompt
        
        Args:
            prompt (str): Text description of desired image
            model (str): Model to use ('stable_diffusion', 'sdxl', 'dreamshaper')
            negative_prompt (str): What to avoid in generation
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow prompt
            width (int): Image width
            height (int): Image height
            seed (int): Random seed for reproducibility
            num_images (int): Number of images to generate
        
        Returns:
            list: Generated PIL Image objects
        """
        
        # Validate prompt
        prompt = validate_prompt(prompt)
        
        # Load model if not already loaded
        pipe = self.model_loader.load_model(model)
        if pipe is None:
            raise ValueError(f"Failed to load model: {model}")
        
        # Set default parameters
        if num_inference_steps is None:
            num_inference_steps = self.config['generation']['default_steps']
        if guidance_scale is None:
            guidance_scale = self.config['generation']['default_guidance_scale']
        
        # Set dimensions based on model
        if width is None or height is None:
            if 'sdxl' in model.lower():
                width = self.config['generation']['sdxl_width']
                height = self.config['generation']['sdxl_height']
            else:
                width = self.config['generation']['default_width']
                height = self.config['generation']['default_height']
        
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.model_loader.device).manual_seed(seed)
        else:
            generator = None
        
        # Print generation info
        print(f"\n{'='*60}")
        print(f"GENERATING IMAGE")
        print(f"{'='*60}")
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        if negative_prompt:
            print(f"Negative: {negative_prompt}")
        print(f"Steps: {num_inference_steps} | Guidance: {guidance_scale}")
        print(f"Size: {width}x{height}")
        if seed:
            print(f"Seed: {seed}")
        print(f"{'='*60}\n")
        
        # Generate images
        generated_images = []
        
        for i in range(num_images):
            if num_images > 1:
                print(f"Generating image {i+1}/{num_images}...")
            
            try:
                # Generate
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    num_images_per_prompt=1
                )
                
                image = result.images[0]
                generated_images.append(image)
                
                # Save image with metadata
                metadata = {
                    'model': model,
                    'negative_prompt': negative_prompt,
                    'steps': num_inference_steps,
                    'guidance_scale': guidance_scale,
                    'width': width,
                    'height': height,
                    'seed': seed
                }
                
                filepath = save_image_with_metadata(
                    image, 
                    prompt, 
                    self.output_path, 
                    metadata
                )
                
                print(f"✓ Image {i+1} generated successfully!")
                
            except Exception as e:
                print(f"✗ Error generating image {i+1}: {e}")
                continue
        
        return generated_images
    
    def generate_batch(self, prompts, model="stable_diffusion", **kwargs):
        """
        Generate multiple images from a list of prompts
        
        Args:
            prompts (list): List of text prompts
            model (str): Model to use
            **kwargs: Additional generation parameters
        
        Returns:
            dict: Dictionary mapping prompts to generated images
        """
        results = {}
        
        print(f"\n{'='*60}")
        print(f"BATCH GENERATION: {len(prompts)} prompts")
        print(f"{'='*60}\n")
        
        for idx, prompt in enumerate(prompts, 1):
            print(f"\n[{idx}/{len(prompts)}] Processing: {prompt[:50]}...")
            
            try:
                images = self.generate(prompt, model=model, **kwargs)
                results[prompt] = images
            except Exception as e:
                print(f"✗ Failed: {e}")
                results[prompt] = None
        
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE: {len([r for r in results.values() if r])} successful")
        print(f"{'='*60}\n")
        
        return results
    
    def generate_with_controlnet(
        self,
        prompt,
        control_image,
        controlnet_conditioning_scale=1.0,
        **kwargs
    ):
        """
        Generate image using ControlNet for better control
        
        Args:
            prompt (str): Text prompt
            control_image (PIL.Image or str): Control image (e.g., canny edges)
            controlnet_conditioning_scale (float): ControlNet influence strength
            **kwargs: Additional generation parameters
        
        Returns:
            PIL.Image: Generated image
        """
        
        # Load control image if path provided
        if isinstance(control_image, str):
            control_image = Image.open(control_image)
        
        # Load ControlNet model
        pipe = self.model_loader.load_controlnet()
        if pipe is None:
            raise ValueError("Failed to load ControlNet model")
        
        print(f"\n{'='*60}")
        print(f"GENERATING WITH CONTROLNET")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Conditioning scale: {controlnet_conditioning_scale}")
        print(f"{'='*60}\n")
        
        # Generate
        try:
            result = pipe(
                prompt=prompt,
                image=control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                **kwargs
            )
            
            image = result.images[0]
            
            # Save
            filepath = save_image_with_metadata(
                image,
                f"[ControlNet] {prompt}",
                self.output_path,
                {'controlnet_scale': controlnet_conditioning_scale}
            )
            
            print("✓ ControlNet generation successful!")
            return image
            
        except Exception as e:
            print(f"✗ Error in ControlNet generation: {e}")
            return None
    
    def compare_models(self, prompt, models=None, **kwargs):
        """
        Generate same prompt with different models for comparison
        
        Args:
            prompt (str): Text prompt
            models (list): List of model names to compare
            **kwargs: Additional generation parameters
        
        Returns:
            dict: Dictionary of model names to generated images
        """
        if models is None:
            models = ['stable_diffusion', 'sdxl', 'dreamshaper']
        
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Models: {', '.join(models)}")
        print(f"{'='*60}\n")
        
        results = {}
        
        for model in models:
            print(f"\nGenerating with {model}...")
            try:
                images = self.generate(prompt, model=model, num_images=1, **kwargs)
                if images:
                    results[model] = images[0]
                    print(f"✓ {model} complete")
            except Exception as e:
                print(f"✗ {model} failed: {e}")
                results[model] = None
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize generator
    generator = ImageGenerator()
    
    # Test prompts
    test_prompts = [
        "A serene mountain landscape at sunset, digital art",
        "A futuristic city with flying cars, cyberpunk style",
        "A cute cat wearing a wizard hat, fantasy art"
    ]
    
    print("Testing Image Generator...")
    print("="*60)
    
    # Generate a single image
    prompt = test_prompts[0]
    images = generator.generate(
        prompt=prompt,
        model="stable_diffusion",
        num_inference_steps=30,
        seed=42
    )
    
    print("\n✓ Test complete!")