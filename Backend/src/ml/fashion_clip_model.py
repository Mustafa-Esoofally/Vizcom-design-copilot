from typing import List, Dict, Any
import torch
from pathlib import Path
from PIL import Image

from src.config.settings import DEVICE, MODEL_NAME, CACHE_DIR

class FashionClipModel:
    def __init__(self):
        self.device = DEVICE
        self.model_name = MODEL_NAME
        self.model = None
        self.processor = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Fashion-CLIP model and processor"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Fashion-CLIP model: {str(e)}")

    def process_image(self, image_path: Path) -> torch.Tensor:
        """Process an image and return its features"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            return image_features
        except Exception as e:
            raise RuntimeError(f"Failed to process image: {str(e)}")

    def process_text(self, text: str) -> torch.Tensor:
        """Process text and return its features"""
        try:
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            text_features = self.model.get_text_features(**inputs)
            return text_features
        except Exception as e:
            raise RuntimeError(f"Failed to process text: {str(e)}")

    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> float:
        """Compute similarity between image and text features"""
        with torch.no_grad():
            similarity = torch.nn.functional.cosine_similarity(
                image_features, text_features
            ).item()
        return similarity

    def classify_image(self, image_path: Path, categories: List[str]) -> Dict[str, float]:
        """Classify an image against a list of text categories"""
        try:
            image_features = self.process_image(image_path)
            results = {}
            
            for category in categories:
                text_features = self.process_text(category)
                similarity = self.compute_similarity(image_features, text_features)
                results[category] = similarity
                
            return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            raise RuntimeError(f"Failed to classify image: {str(e)}")

    def __call__(self, image_path: Path, query: str) -> float:
        """Compute similarity between an image and a text query"""
        image_features = self.process_image(image_path)
        text_features = self.process_text(query)
        return self.compute_similarity(image_features, text_features) 