\# Quick Start Guide - Image Generation System



\## 🚀 Getting Started in 5 Minutes



\### Step 1: Setup (One-time only)

```powershell

\# In PowerShell, navigate to your project folder

cd E:\\PROJECTS



\# Activate your virtual environment

.\\LLM\\Scripts\\Activate.ps1



\# Run the setup script

.\\setup.ps1

```



This will:

\- ✅ Check your system

\- ✅ Install all dependencies

\- ✅ Create necessary folders

\- ✅ Verify everything works



\*\*Note:\*\* First setup takes 5-10 minutes to download packages.



---



\## 🎨 Generate Your First Image



\### Option 1: Web Interface (Easiest!)

```powershell

python gradio\_ui.py

```

Then open: http://127.0.0.1:7860



---



\### Option 2: Interactive Mode

```powershell

python main.py

```



Then type:

```

generate "a beautiful sunset over mountains" --model stable\_diffusion

```



---



\### Option 3: Command Line

```powershell

python main.py generate "a cute robot in a garden" --model stable\_diffusion --steps 30

```



---



\## 📝 Example Prompts to Try



\*\*Landscapes:\*\*

```

A serene mountain landscape at sunset, digital art, highly detailed, vibrant colors

```



\*\*Characters:\*\*

```

A cute robot playing with a cat, studio lighting, 3D render, high quality

```



\*\*Fantasy:\*\*

```

An enchanted forest with glowing mushrooms, fantasy art, magical atmosphere, detailed

```



\*\*Sci-Fi:\*\*

```

A futuristic cityscape at night, cyberpunk style, neon lights, rain reflections

```



---



\## 🎛️ Common Commands



\### Generate with Different Models

```powershell

\# Stable Diffusion (Fast, good quality)

python main.py generate "your prompt" --model stable\_diffusion



\# SDXL (Slower, best quality)

python main.py generate "your prompt" --model sdxl



\# DreamShaper (Artistic style)

python main.py generate "your prompt" --model dreamshaper

```



\### Compare Models

```powershell

python main.py compare "a magical forest" --evaluate

```



\### Batch Generation

```powershell

python main.py batch --prompts "cat;dog;bird" --model stable\_diffusion

```



---



\## ⚙️ Important Parameters



| Parameter | What it does | Recommended |

|-----------|-------------|-------------|

| `--steps` | Quality vs Speed | 30-50 |

| `--guidance` | Follow prompt closely | 7-9 |

| `--seed` | Reproducibility | Any number |

| `--model` | Which AI to use | stable\_diffusion |



\*\*Example with parameters:\*\*

```powershell

python main.py generate "a dragon" --steps 40 --guidance 8 --seed 42

```



---



\## 📂 Where Are My Images?



All generated images are saved in:

```

E:\\PROJECTS\\data\\generated\_images\\

```



Each image comes with:

\- The image file (PNG)

\- A metadata file (JSON) with generation details



---



\## 🐛 Troubleshooting



\### "Out of memory" error

\*\*Solution 1:\*\* Use a smaller model

```powershell

python main.py generate "prompt" --model stable\_diffusion

```



\*\*Solution 2:\*\* Edit `config.yaml`:

```yaml

hardware:

&nbsp; enable\_cpu\_offload: true

```



\### "CUDA not available"

\- Don't worry! It will use CPU (just slower)

\- Or install GPU drivers from NVIDIA



\### Models downloading slowly

\- First time only - models are 2-8GB

\- Once downloaded, generation is fast



---



\## 💡 Tips for Best Results



1\. \*\*Be Specific\*\*

&nbsp;  - ❌ "a cat"

&nbsp;  - ✅ "a fluffy orange cat sitting in a garden, sunlight, detailed fur, professional photo"



2\. \*\*Add Style Keywords\*\*

&nbsp;  - "digital art", "oil painting", "3D render", "photograph"

&nbsp;  - "highly detailed", "professional", "studio lighting"



3\. \*\*Use Negative Prompts\*\* (Web UI or via code)

&nbsp;  - "blurry, low quality, distorted, ugly, bad anatomy"



4\. \*\*Experiment with Settings\*\*

&nbsp;  - Start with 30 steps

&nbsp;  - Try guidance 7-9

&nbsp;  - Use same seed to refine



---



\## 📖 Next Steps



1\. ✅ Generate your first image

2\. ✅ Try different models

3\. ✅ Experiment with prompts

4\. ✅ Read full README.md for advanced features



---



\## 🆘 Need Help?



Check these in order:

1\. This Quick Start Guide

2\. README.md (detailed documentation)

3\. Run: `python main.py --help`



---



\## 🎉 You're Ready!



Start creating amazing images with AI!



\*\*Quick command to get started:\*\*

```powershell

python gradio\_ui.py

```



Then click "Generate Image" and watch the magic happen! ✨

