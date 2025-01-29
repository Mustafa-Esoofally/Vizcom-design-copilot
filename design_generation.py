import torch
from diffusers import StableDiffusionXLPipeline, ControlNetModel
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional
import os
import json
from functools import lru_cache

class DesignGenerationPipeline:
    def __init__(self, model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        """Initialize the design generation pipeline."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Initialize SDXL model
        self.pipeline = self._load_pipeline(model_path)
        
        # Load ControlNet for style control
        self.controlnet = self._load_controlnet()
    
    @lru_cache(maxsize=1)
    def _load_pipeline(self, model_path: str) -> StableDiffusionXLPipeline:
        """Load and cache the SDXL pipeline."""
        return StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            variant="fp16" if self.device == "cuda" else None,
            use_safetensors=True
        ).to(self.device)
    
    @lru_cache(maxsize=1)
    def _load_controlnet(self) -> Optional[ControlNetModel]:
        """Load and cache the ControlNet model."""
        try:
            return ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_openpose",
                torch_dtype=self.dtype
            ).to(self.device)
        except Exception as e:
            print(f"Warning: Failed to load ControlNet: {e}")
            return None
        
    def generate_designs(self, design_brief: Dict[str, Any], 
                        reference_images: List[str],
                        num_variations: int = 4) -> List[Dict[str, Any]]:
        """Generate design variations based on brief and references."""
        try:
            # Prepare prompts
            prompts = self._prepare_prompts(design_brief)
            
            # Generate base designs
            designs = []
            for prompt in prompts:
                # Prepare control image if available
                control_image = None
                if reference_images and self.controlnet:
                    control_image = self._prepare_control_image(reference_images[0])
                
                # Generate variations
                variations = self._generate_variations(
                    prompt=prompt,
                    control_image=control_image,
                    num_variations=num_variations
                )
                designs.extend(variations)
            
            return designs
            
        except Exception as e:
            print(f"Error generating designs: {e}")
            return []
    
    def _prepare_prompts(self, design_brief: Dict[str, Any]) -> List[str]:
        """Prepare generation prompts from design brief."""
        style = design_brief.get('style', '')
        colors = design_brief.get('colors', [])
        silhouette = design_brief.get('silhouette', '')
        
        prompts = []
        base_prompt = f"Fashion design, {style} style"
        
        if colors:
            for color in colors[:2]:  # Use top 2 colors
                prompt = f"{base_prompt}, {color} color"
                if silhouette:
                    prompt += f", {silhouette} silhouette"
                prompt += ", highly detailed, professional fashion photography"
                prompts.append(prompt)
        else:
            prompts.append(f"{base_prompt}, highly detailed, professional fashion photography")
        
        return prompts
    
    def _generate_variations(self, prompt: str, control_image: Optional[Image.Image] = None,
                           num_variations: int = 4) -> List[Dict[str, Any]]:
        """Generate design variations using SDXL."""
        variations = []
        
        # Set up generation parameters
        params = {
            "prompt": prompt,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "negative_prompt": "low quality, blurry, distorted"
        }
        
        if control_image and self.controlnet:
            params["image"] = control_image
        
        # Generate variations
        for _ in range(num_variations):
            try:
                image = self.pipeline(**params).images[0]
                variations.append({
                    'image': image,
                    'metadata': {
                        'prompt': prompt,
                        'parameters': {
                            k: v for k, v in params.items() 
                            if k not in ['image', 'prompt']
                        }
                    }
                })
            except Exception as e:
                print(f"Error generating variation: {e}")
                continue
        
        return variations
    
    def _prepare_control_image(self, image_path: str) -> Optional[Image.Image]:
        """Prepare control image for ControlNet."""
        try:
            image = Image.open(image_path)
            # Resize to SDXL's preferred size
            target_size = (1024, 1024)
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f"Error preparing control image: {e}")
            return None
    
    def save_designs(self, designs: List[Dict[str, Any]], output_dir: str):
        """Save generated designs to output directory."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for idx, design in enumerate(designs):
                # Save image
                image_path = os.path.join(output_dir, f"design_{idx}.png")
                design['image'].save(image_path)
                
                # Save metadata
                meta_path = os.path.join(output_dir, f"design_{idx}_meta.json")
                with open(meta_path, 'w') as f:
                    json.dump(design['metadata'], f, indent=2)
                    
        except Exception as e:
            print(f"Error saving designs: {e}")

def main():
    """Example usage of the design generation pipeline."""
    try:
        pipeline = DesignGenerationPipeline()
        
        # Example design brief
        brief = {
            'style': 'modern minimalist',
            'colors': ['navy', 'grey'],
            'silhouette': 'relaxed'
        }
        
        # Example reference images
        references = ['reference1.jpg', 'reference2.jpg']
        
        # Generate designs
        designs = pipeline.generate_designs(brief, references)
        
        # Save results
        if designs:
            pipeline.save_designs(designs, 'generated_designs')
            print(f"Successfully generated and saved {len(designs)} designs")
        else:
            print("No designs were generated")
            
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main() 