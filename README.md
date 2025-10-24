#  AI Image Generation System

A comprehensive multi-model text-to-image generation system powered by state-of-the-art AI models including Stable Diffusion, SDXL, DreamShaper, and ControlNet.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

##  Features

-  **Multiple AI Models**: Stable Diffusion v1.5, SDXL, DreamShaper, ControlNet
-  **Three Interfaces**: Web UI (Gradio), Interactive CLI, Direct commands
-  **Quality Evaluation**: CLIP scores, sharpness, contrast analysis
-  **GPU Accelerated**: Fast generation (10-30 seconds per image)
-  **Model Comparison**: Side-by-side comparison of different models
-  **Batch Processing**: Generate multiple images efficiently
-  **Fine Control**: Adjust steps, guidance scale, resolution, seeds

##  Quick Start

\\\ash
# Clone the repository
git clone https://github.com/amal-hash-tes/ai-image-generation-system.git
cd ai-image-generation-system

# Install dependencies
pip install -r requirements.txt

# Launch web interface
python gradio_ui.py
\\\

Then open: http://127.0.0.1:7860

##  Requirements

- Python 3.8+
- NVIDIA GPU with 6GB+ VRAM (recommended)
- 16GB RAM
- 20GB free disk space

##  Example Usage

### Web Interface
\\\ash
python gradio_ui.py
\\\

### Command Line
\\\ash
python main.py generate "a serene mountain landscape at sunset" --model stable_diffusion
\\\

### Interactive Mode
\\\ash
python main.py
# Then type: generate "your prompt" --model sdxl
\\\

##  Supported Models

| Model | Resolution | VRAM | Speed | Best For |
|-------|-----------|------|-------|----------|
| **Stable Diffusion v1.5** | 512x512 | ~4GB | Fast | General purpose |
| **SDXL** | 1024x1024 | ~8GB | Slower | Highest quality |
| **DreamShaper** | 512x512 | ~4GB | Fast | Artistic style |
| **ControlNet** | 512x512 | ~6GB | Moderate | Precise control |

##  Evaluation Metrics

- **CLIP Score**: Measures prompt-image relevance (0-100)
- **Sharpness**: Image detail and clarity
- **Contrast**: Dynamic range analysis
- **Brightness**: Overall luminosity

##  Tech Stack

- **Python** - Core language
- **PyTorch** - Deep learning framework
- **Diffusers** - Hugging Face diffusion models
- **Transformers** - Text encoding
- **Gradio** - Web interface
- **CLIP** - Image-text evaluation

##  Project Structure

\\\
image-generation-system/
 main.py                 # CLI interface
 gradio_ui.py           # Web interface
 image_generator.py     # Core generation logic
 model_loader.py        # Model management
 evaluation.py          # Quality assessment
 utils.py               # Helper functions
 config.yaml            # Configuration
 requirements.txt       # Dependencies
 data/
     generated_images/  # Output directory
\\\

##  Sample Prompts

\\\
"A serene mountain landscape at sunset, digital art, highly detailed"
"A futuristic cityscape at night, cyberpunk style, neon lights"
"A cute robot playing with a cat in a garden, 3D render"
"An enchanted forest with glowing mushrooms, fantasy art"
\\\

##  Documentation

- [Installation Guide](INSTALLATION.md) - Detailed setup instructions
- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- [Full Documentation](README.md) - Complete feature documentation

##  Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

##  License

MIT License - feel free to use this project for learning and development.

##  Acknowledgments

- [Stability AI](https://stability.ai/) - Stable Diffusion models
- [Hugging Face](https://huggingface.co/) - Model hosting and libraries
- [OpenAI](https://openai.com/) - CLIP model

##  Contact

**Amal** - [GitHub](https://github.com/amal-hash-tes)

---

 Star this repo if you find it helpful!
