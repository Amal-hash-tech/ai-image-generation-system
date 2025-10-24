"""
Gradio Web Interface for Image Generation System
Provides an easy-to-use web UI for generating and evaluating images
"""

import gradio as gr
from image_generator import ImageGenerator
from evaluation import ImageEvaluator
from model_loader import ModelLoader
import torch


class ImageGenerationUI:
    """Web UI for image generation"""
    
    def __init__(self):
        """Initialize the UI"""
        self.generator = ImageGenerator()
        self.evaluator = ImageEvaluator()
        
        print("✓ UI initialized successfully!")
    
    def generate_image(
        self,
        prompt,
        model,
        negative_prompt,
        steps,
        guidance_scale,
        width,
        height,
        seed,
        evaluate
    ):
        """Generate image from UI inputs"""
        
        try:
            # Handle seed
            if seed == -1:
                seed = None
            
            # Generate image
            images = self.generator.generate(
                prompt=prompt,
                model=model.lower().replace(" ", "_"),
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed,
                num_images=1
            )
            
            if not images:
                return None, "❌ Generation failed"
            
            image = images[0]
            
            # Evaluate if requested
            if evaluate:
                eval_results = self.evaluator.evaluate_image(image, prompt)
                
                results_text = f"""
                **Evaluation Results:**
                - CLIP Score: {eval_results.get('clip_score', 'N/A')}
                - Sharpness: {eval_results.get('sharpness', 'N/A'):.2f}
                - Contrast: {eval_results.get('contrast', 'N/A'):.2f}
                - Brightness: {eval_results.get('brightness', 'N/A'):.2f}
                - Resolution: {eval_results.get('resolution', 'N/A')}
                """
            else:
                results_text = "✅ Image generated successfully!"
            
            return image, results_text
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def compare_models(self, prompt, steps, guidance_scale):
        """Compare different models"""
        
        try:
            models = ['stable_diffusion', 'sdxl', 'dreamshaper']
            results = self.generator.compare_models(
                prompt=prompt,
                models=models,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            )
            
            # Evaluate all models
            model_results = {model: (img, prompt) for model, img in results.items() if img}
            comparison = self.evaluator.compare_models_evaluation(model_results)
            
            # Format results
            output_images = []
            output_text = "**Model Comparison:**\n\n"
            
            for model, img in results.items():
                if img:
                    output_images.append((img, model))
                    eval_data = comparison.get(model, {})
                    output_text += f"**{model}:**\n"
                    output_text += f"- CLIP Score: {eval_data.get('clip_score', 'N/A')}\n"
                    output_text += f"- Sharpness: {eval_data.get('sharpness', 'N/A'):.2f}\n\n"
            
            return output_images, output_text
            
        except Exception as e:
            return [], f"❌ Error: {str(e)}"
    
    def launch(self):
        """Launch the Gradio interface"""
        
        # Custom CSS
        custom_css = """
        .container {
            max-width: 1200px;
            margin: auto;
        }
        .output-image {
            max-height: 600px;
        }
        """
        
        with gr.Blocks(css=custom_css, title="Image Generation System") as demo:
            gr.Markdown("""
            # 🎨 Image Generation System
            ### Generate high-quality images from text prompts using multiple AI models
            """)
            
            with gr.Tabs():
                # Tab 1: Single Generation
                with gr.Tab("Generate Image"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            prompt_input = gr.Textbox(
                                label="Prompt",
                                placeholder="A serene mountain landscape at sunset...",
                                lines=3
                            )
                            negative_prompt_input = gr.Textbox(
                                label="Negative Prompt (Optional)",
                                placeholder="blurry, low quality, distorted...",
                                lines=2
                            )
                            
                            model_dropdown = gr.Dropdown(
                                choices=["Stable Diffusion", "SDXL", "DreamShaper"],
                                value="Stable Diffusion",
                                label="Model"
                            )
                            
                            with gr.Row():
                                steps_slider = gr.Slider(
                                    minimum=10,
                                    maximum=100,
                                    value=50,
                                    step=5,
                                    label="Inference Steps"
                                )
                                guidance_slider = gr.Slider(
                                    minimum=1,
                                    maximum=20,
                                    value=7.5,
                                    step=0.5,
                                    label="Guidance Scale"
                                )
                            
                            with gr.Row():
                                width_slider = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=64,
                                    label="Width"
                                )
                                height_slider = gr.Slider(
                                    minimum=256,
                                    maximum=1024,
                                    value=512,
                                    step=64,
                                    label="Height"
                                )
                            
                            seed_input = gr.Number(
                                label="Seed (-1 for random)",
                                value=-1
                            )
                            
                            evaluate_checkbox = gr.Checkbox(
                                label="Evaluate generated image",
                                value=True
                            )
                            
                            generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                        
                        with gr.Column(scale=1):
                            output_image = gr.Image(label="Generated Image", type="pil")
                            output_text = gr.Markdown()
                    
                    # Connect button
                    generate_btn.click(
                        fn=self.generate_image,
                        inputs=[
                            prompt_input,
                            model_dropdown,
                            negative_prompt_input,
                            steps_slider,
                            guidance_slider,
                            width_slider,
                            height_slider,
                            seed_input,
                            evaluate_checkbox
                        ],
                        outputs=[output_image, output_text]
                    )
                
                # Tab 2: Model Comparison
                with gr.Tab("Compare Models"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            compare_prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="A futuristic city at night...",
                                lines=3
                            )
                            
                            compare_steps = gr.Slider(
                                minimum=10,
                                maximum=50,
                                value=30,
                                step=5,
                                label="Inference Steps"
                            )
                            
                            compare_guidance = gr.Slider(
                                minimum=1,
                                maximum=15,
                                value=7.5,
                                step=0.5,
                                label="Guidance Scale"
                            )
                            
                            compare_btn = gr.Button("🔄 Compare Models", variant="primary")
                        
                        with gr.Column(scale=2):
                            compare_output = gr.Gallery(
                                label="Model Comparison",
                                columns=3,
                                rows=1,
                                object_fit="contain"
                            )
                            compare_text = gr.Markdown()
                    
                    compare_btn.click(
                        fn=self.compare_models,
                        inputs=[compare_prompt, compare_steps, compare_guidance],
                        outputs=[compare_output, compare_text]
                    )
                
                # Tab 3: Information
                with gr.Tab("ℹ️ Information"):
                    gr.Markdown("""
                    ## About
                    
                    This system supports multiple state-of-the-art text-to-image models:
                    
                    ### Available Models
                    
                    **🎨 Stable Diffusion v1.5**
                    - Fast and efficient
                    - Best for general use
                    - 512x512 default resolution
                    - Lower VRAM requirements (~4GB)
                    
                    **✨ Stable Diffusion XL (SDXL)**
                    - Highest quality output
                    - Better detail and composition
                    - 1024x1024 default resolution
                    - Higher VRAM requirements (~8GB)
                    
                    **🌟 DreamShaper**
                    - Artistic and creative
                    - Vibrant colors
                    - Good for fantasy/artistic content
                    - Similar requirements to SD v1.5
                    
                    ### Tips for Best Results
                    
                    1. **Be Specific**: Include details about style, mood, lighting, and composition
                    2. **Use Negative Prompts**: Specify what you don't want in the image
                    3. **Adjust Steps**: More steps = better quality but slower generation
                    4. **Guidance Scale**: Higher values follow prompt more closely (7-9 recommended)
                    5. **Seed**: Use the same seed to reproduce similar results
                    
                    ### Evaluation Metrics
                    
                    - **CLIP Score**: Measures how well image matches prompt (higher is better)
                    - **Sharpness**: Image detail and clarity
                    - **Contrast**: Dynamic range of the image
                    - **Brightness**: Overall luminosity
                    
                    """)
                    
                    # System info
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    if device.type == "cuda":
                        gpu_name = torch.cuda.get_device_name(0)
                        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        system_info = f"""
                        ### System Information
                        - Device: {gpu_name}
                        - VRAM: {vram:.2f} GB
                        - Mode: GPU Accelerated ⚡
                        """
                    else:
                        system_info = """
                        ### System Information
                        - Device: CPU
                        - Mode: CPU (slower generation)
                        """
                    
                    gr.Markdown(system_info)
            
            # Examples
            gr.Examples(
                examples=[
                    ["A serene mountain landscape at sunset, digital art, highly detailed", "Stable Diffusion"],
                    ["A cute robot playing with a cat, studio lighting, 3D render", "SDXL"],
                    ["An enchanted forest with glowing mushrooms, fantasy art, magical atmosphere", "DreamShaper"],
                    ["A futuristic cityscape at night, cyberpunk style, neon lights", "Stable Diffusion"],
                ],
                inputs=[prompt_input, model_dropdown],
            )
        
        print("\n" + "="*60)
        print("Starting Gradio Interface...")
        print("="*60 + "\n")
        
        # Launch
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860
        )


def main():
    """Main function to launch UI"""
    ui = ImageGenerationUI()
    ui.launch()


if __name__ == "__main__":
    main()