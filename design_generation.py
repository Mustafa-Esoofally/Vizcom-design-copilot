import torch
from diffusers import StableDiffusionXLPipeline, ControlNetModel
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import os
import json

class DesignGenerationPipeline:
    def __init__(self, model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        """Initialize the design generation pipeline."""
        # Initialize SDXL model
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        # Load ControlNet for style control
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16
        ).to("cuda")
        
        # Initialize style transfer model
        self.style_transfer = self.load_style_transfer_model()
        
    def generate_designs(self, design_brief: Dict[str, Any], 
                        reference_images: List[str],
                        num_variations: int = 4) -> List[Dict[str, Any]]:
        """Generate design variations based on brief and references."""
        # Prepare prompts from brief
        prompts = self.prepare_prompts(design_brief)
        
        # Generate base designs
        base_designs = []
        for prompt in prompts:
            variations = self.generate_variations(
                prompt,
                reference_images,
                num_variations
            )
            base_designs.extend(variations)
        
        # Apply style transfer
        styled_designs = []
        for design in base_designs:
            styled = self.apply_style_transfer(
                design['image'],
                reference_images[0]  # Use first reference as style guide
            )
            styled_designs.append({
                'image': styled,
                'prompt': design['prompt'],
                'style_score': self.evaluate_style(styled, reference_images)
            })
        
        return styled_designs
    
    def prepare_prompts(self, design_brief: Dict[str, Any]) -> List[str]:
        """Prepare generation prompts from design brief."""
        prompts = []
        
        # Extract key elements from brief
        theme = design_brief['design_direction']['primary_theme']
        elements = design_brief['style_attributes']['design_elements']
        materials = design_brief['style_attributes']['materials']
        colors = design_brief['style_attributes']['color_palette']
        
        # Create base prompt template
        base_prompt = f"Fashion design, {theme}, "
        
        # Generate variations focusing on different aspects
        for element in elements[:2]:
            for material in materials[:2]:
                for color in colors[:2]:
                    prompt = base_prompt + f"{element}, {material}, {color}, "
                    prompt += "highly detailed, professional fashion photography"
                    prompts.append(prompt)
        
        return prompts
    
    def generate_variations(self, prompt: str, reference_images: List[str], 
                          num_variations: int) -> List[Dict[str, Any]]:
        """Generate design variations using SDXL."""
        # Prepare reference image for ControlNet
        control_image = self.prepare_control_image(reference_images[0])
        
        # Generate variations
        variations = []
        for _ in range(num_variations):
            image = self.pipeline(
                prompt=prompt,
                image=control_image,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
            
            variations.append({
                'image': image,
                'prompt': prompt
            })
        
        return variations
    
    def prepare_control_image(self, image_path: str) -> Image.Image:
        """Prepare reference image for ControlNet guidance."""
        # Load and preprocess image
        image = Image.open(image_path)
        image = image.resize((1024, 1024))
        return image
    
    def load_style_transfer_model(self):
        """Load AdaIN++ style transfer model."""
        # Placeholder for style transfer model initialization
        class StyleTransfer:
            def __call__(self, content_image, style_image):
                # Placeholder for style transfer
                return content_image
        
        return StyleTransfer()
    
    def apply_style_transfer(self, content_image: Image.Image, 
                           style_image_path: str) -> Image.Image:
        """Apply style transfer to generated design."""
        # Load style image
        style_image = Image.open(style_image_path).convert('RGB')
        style_image = style_image.resize(content_image.size)
        
        # Apply style transfer
        styled_image = self.style_transfer(content_image, style_image)
        return styled_image
    
    def evaluate_style(self, generated_image: Image.Image, 
                      reference_images: List[str]) -> float:
        """Evaluate style consistency with reference images."""
        # Placeholder for style evaluation logic
        return 0.85
    
    def save_designs(self, designs: List[Dict[str, Any]], output_dir: str):
        """Save generated designs with metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, design in enumerate(designs):
            # Save image
            image_path = os.path.join(output_dir, f"design_{i}.png")
            design['image'].save(image_path)
            
            # Save metadata
            metadata_path = os.path.join(output_dir, f"design_{i}_metadata.json")
            metadata = {
                'prompt': design['prompt'],
                'style_score': design['style_score']
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

def main():
    # Example usage
    pipeline = DesignGenerationPipeline()
    
    # Sample design brief
    design_brief = {
        'design_direction': {
            'primary_theme': 'urban performance'
        },
        'style_attributes': {
            'design_elements': ['asymmetric zip', 'reflective details'],
            'materials': ['technical fabric', 'performance mesh'],
            'color_palette': ['navy', 'charcoal', 'red']
        }
    }
    
    # Sample reference images
    reference_images = ['reference1.jpg', 'reference2.jpg']
    
    # Generate designs
    designs = pipeline.generate_designs(design_brief, reference_images)
    
    # Save results
    pipeline.save_designs(designs, 'output_designs')

if __name__ == "__main__":
    main() 