"""
Evaluation Module - Assess image quality and prompt-image relevance
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from utils import load_config
import os


class ImageEvaluator:
    """Evaluate generated images using various metrics"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize evaluator with CLIP model"""
        self.config = load_config(config_path)
        
        print("\n📊 Initializing Image Evaluator...")
        
        # Load CLIP for text-image similarity
        clip_model_name = self.config['evaluation']['clip_model']
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = self.clip_model.to(self.device)
        
        print(f"✓ CLIP model loaded: {clip_model_name}")
        print(f"✓ Device: {self.device}\n")
    
    def calculate_clip_score(self, image, prompt):
        """
        Calculate CLIP score (text-image similarity)
        Higher score means better alignment between prompt and image
        
        Args:
            image (PIL.Image): Generated image
            prompt (str): Text prompt used for generation
        
        Returns:
            float: CLIP score (0-100)
        """
        
        # Prepare inputs
        inputs = self.clip_processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Calculate similarity
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.item()
        
        # Convert to 0-100 scale
        clip_score = (score / 100) * 100
        
        return clip_score
    
    def evaluate_image(self, image, prompt):
        """
        Comprehensive evaluation of a single image
        
        Args:
            image (PIL.Image or str): Image to evaluate
            prompt (str): Original prompt
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image)
        
        print(f"\n{'='*60}")
        print(f"EVALUATING IMAGE")
        print(f"{'='*60}")
        print(f"Prompt: {prompt[:60]}...")
        
        results = {}
        
        # 1. CLIP Score (prompt-image relevance)
        try:
            clip_score = self.calculate_clip_score(image, prompt)
            results['clip_score'] = round(clip_score, 2)
            print(f"✓ CLIP Score: {results['clip_score']:.2f}")
        except Exception as e:
            print(f"✗ CLIP Score failed: {e}")
            results['clip_score'] = None
        
        # 2. Image Quality Metrics
        try:
            quality_metrics = self.calculate_quality_metrics(image)
            results.update(quality_metrics)
            print(f"✓ Sharpness: {quality_metrics['sharpness']:.2f}")
            print(f"✓ Contrast: {quality_metrics['contrast']:.2f}")
            print(f"✓ Brightness: {quality_metrics['brightness']:.2f}")
        except Exception as e:
            print(f"✗ Quality metrics failed: {e}")
        
        # 3. Resolution
        results['resolution'] = f"{image.width}x{image.height}"
        results['aspect_ratio'] = round(image.width / image.height, 2)
        
        print(f"✓ Resolution: {results['resolution']}")
        print(f"{'='*60}\n")
        
        return results
    
    def calculate_quality_metrics(self, image):
        """
        Calculate basic image quality metrics
        
        Args:
            image (PIL.Image): Image to analyze
        
        Returns:
            dict: Quality metrics
        """
        
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Calculate metrics
        metrics = {}
        
        # 1. Sharpness (using Laplacian variance)
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)
        laplacian = ndimage.laplace(gray)
        metrics['sharpness'] = float(np.var(laplacian))
        
        # 2. Contrast (standard deviation of pixel values)
        metrics['contrast'] = float(np.std(img_array))
        
        # 3. Brightness (mean pixel value)
        metrics['brightness'] = float(np.mean(img_array))
        
        # 4. Color diversity (unique colors)
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        metrics['color_diversity'] = unique_colors
        
        return metrics
    
    def evaluate_batch(self, images_with_prompts):
        """
        Evaluate multiple images
        
        Args:
            images_with_prompts (list): List of (image, prompt) tuples
        
        Returns:
            list: List of evaluation results
        """
        
        print(f"\n{'='*60}")
        print(f"BATCH EVALUATION: {len(images_with_prompts)} images")
        print(f"{'='*60}\n")
        
        results = []
        
        for idx, (image, prompt) in enumerate(images_with_prompts, 1):
            print(f"[{idx}/{len(images_with_prompts)}] Evaluating...")
            
            try:
                eval_result = self.evaluate_image(image, prompt)
                eval_result['image_index'] = idx
                results.append(eval_result)
            except Exception as e:
                print(f"✗ Evaluation failed: {e}")
                results.append(None)
        
        # Calculate average scores
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """Print summary statistics of evaluation results"""
        
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            print("\n⚠ No valid results to summarize")
            return
        
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        # Average CLIP scores
        clip_scores = [r['clip_score'] for r in valid_results if r.get('clip_score')]
        if clip_scores:
            avg_clip = np.mean(clip_scores)
            print(f"Average CLIP Score: {avg_clip:.2f}")
            print(f"Min CLIP Score: {min(clip_scores):.2f}")
            print(f"Max CLIP Score: {max(clip_scores):.2f}")
        
        # Average quality metrics
        sharpness_scores = [r.get('sharpness', 0) for r in valid_results]
        if sharpness_scores:
            print(f"\nAverage Sharpness: {np.mean(sharpness_scores):.2f}")
        
        contrast_scores = [r.get('contrast', 0) for r in valid_results]
        if contrast_scores:
            print(f"Average Contrast: {np.mean(contrast_scores):.2f}")
        
        print(f"\nTotal Images Evaluated: {len(valid_results)}")
        print(f"{'='*60}\n")
    
    def compare_models_evaluation(self, model_results):
        """
        Compare evaluation results across different models
        
        Args:
            model_results (dict): Dictionary of {model_name: (image, prompt)}
        
        Returns:
            dict: Comparison results
        """
        
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON EVALUATION")
        print(f"{'='*60}\n")
        
        comparison = {}
        
        for model_name, (image, prompt) in model_results.items():
            print(f"Evaluating {model_name}...")
            try:
                eval_result = self.evaluate_image(image, prompt)
                comparison[model_name] = eval_result
            except Exception as e:
                print(f"✗ {model_name} evaluation failed: {e}")
                comparison[model_name] = None
        
        # Print comparison
        print(f"\n{'='*60}")
        print(f"COMPARISON RESULTS")
        print(f"{'='*60}")
        
        for model_name, results in comparison.items():
            if results:
                print(f"\n{model_name}:")
                print(f"  CLIP Score: {results.get('clip_score', 'N/A')}")
                print(f"  Sharpness: {results.get('sharpness', 'N/A'):.2f}")
                print(f"  Contrast: {results.get('contrast', 'N/A'):.2f}")
        
        # Find best model
        valid_models = {k: v for k, v in comparison.items() if v and v.get('clip_score')}
        if valid_models:
            best_model = max(valid_models, key=lambda k: valid_models[k]['clip_score'])
            print(f"\n🏆 Best Model (by CLIP score): {best_model}")
        
        print(f"{'='*60}\n")
        
        return comparison
    
    def calculate_prompt_adherence(self, image, prompt, keywords=None):
        """
        Calculate how well the image adheres to specific keywords in prompt
        
        Args:
            image (PIL.Image): Generated image
            prompt (str): Original prompt
            keywords (list): Specific keywords to check (optional)
        
        Returns:
            dict: Keyword adherence scores
        """
        
        if keywords is None:
            # Extract important words from prompt
            keywords = [word for word in prompt.split() if len(word) > 3]
        
        adherence_scores = {}
        
        for keyword in keywords:
            score = self.calculate_clip_score(image, keyword)
            adherence_scores[keyword] = round(score, 2)
        
        return adherence_scores


# Example usage
if __name__ == "__main__":
    evaluator = ImageEvaluator()
    
    print("Evaluator initialized and ready!")
    print("\nYou can use it to evaluate generated images:")
    print("  evaluator.evaluate_image(image, prompt)")
    print("  evaluator.evaluate_batch([(img1, prompt1), (img2, prompt2)])")