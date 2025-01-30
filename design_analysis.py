from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from PIL import Image
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json
from rdflib import Graph, Namespace, Literal, URIRef
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_models():
    """Load required models from Hugging Face"""
    # Load CLIP for style analysis
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load Grounding DINO for object detection
    processor = AutoProcessor.from_pretrained("ShilongLiu/GroundingDINO-SwinT-OGC")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("ShilongLiu/GroundingDINO-SwinT-OGC")
    
    return {
        'clip': (clip_model, clip_processor),
        'dino': (model, processor)
    }

def analyze_design(image_path, brief):
    """Analyze a single design image"""
    models = load_models()
    clip_model, clip_processor = models['clip']
    dino_model, dino_processor = models['dino']
    
    # Load and process image
    image = Image.open(image_path)
    
    # CLIP Analysis
    inputs = clip_processor(images=image, text=brief, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    similarity = outputs.logits_per_image.item()
    
    # Grounding DINO Analysis
    inputs = dino_processor(images=image, text=brief, return_tensors="pt")
    outputs = dino_model(**inputs)
    
    # Process results
    target_score = outputs.logits.sigmoid().max().item()
    
    return {
        'style_match': similarity,
        'object_detection_score': target_score,
        'image_path': image_path
    }

def process_season_designs(season_path, brief):
    """Process all designs in a season folder"""
    results = []
    for image_file in os.listdir(season_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(season_path, image_file)
            result = analyze_design(image_path, brief)
            results.append(result)
    return results

def main():
    brief = "father suiting pant blue"
    
    # Process both seasons
    season1_results = process_season_designs("Season_1", brief)
    season2_results = process_season_designs("Season_2", brief)
    
    # Combine and sort results
    all_results = season1_results + season2_results
    sorted_results = sorted(all_results, key=lambda x: x['style_match'] * x['object_detection_score'], reverse=True)
    
    # Print top 5 matches
    print("\nTop 5 Design Matches:")
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. {result['image_path']}")
        print(f"   Style Match: {result['style_match']:.2f}")
        print(f"   Object Detection Score: {result['object_detection_score']:.2f}")

class DesignAnalysisAgent:
    def __init__(self, store_path: str = "design_store"):
        self.store_path = store_path
        self.embeddings = HuggingFaceEmbeddings()
        self.store = Chroma(
            persist_directory=store_path,
            embedding_function=self.embeddings
        )
        
    def analyze_brief(self, brief_text: str, reference_images: List[str], brand_guidelines: Dict[str, Any]) -> Dict[str, Any]:
        # Embed brief text
        brief_embedding = self.embeddings.embed_query(brief_text)
        brief_embedding = torch.tensor(brief_embedding)
        
        # Extract features from reference images
        image_features = [
            self.extract_image_features(img_path)
            for img_path in reference_images
        ]
        
        # Store project context
        self.store_project_context(brief_text, image_features, brand_guidelines)
        
        # Extract style attributes
        style_attributes = self.extract_style_attributes(brief_embedding, image_features)
        
        # Build style knowledge graph
        self.build_style_graph(style_attributes, brand_guidelines)
        
        # Get relevant brand context
        brand_context = self.get_brand_context(style_attributes)
        
        # Generate design direction
        design_direction = self.generate_design_direction(brief_embedding, style_attributes)
        
        return {
            'style_attributes': style_attributes,
            'brand_context': brand_context,
            'design_direction': design_direction
        }

    def extract_image_features(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs = processor(images=image, return_tensors="pt")
        features = model.get_image_features(**inputs)
        return features.squeeze()

    def store_project_context(self, brief_text: str, image_features: List[torch.Tensor], 
                            brand_guidelines: Dict[str, Any]):
        # Store brief embedding
        brief_embedding = self.embeddings.embed_query(brief_text)
        self.store.add_texts(
            texts=[brief_text],
            embeddings=[brief_embedding],
            metadatas=[{'type': 'brief'}]
        )
        
        # Store image features
        for idx, features in enumerate(image_features):
            self.store.add_texts(
                texts=[f"image_{idx}"],
                embeddings=[features.tolist()],
                metadatas=[{'type': 'image'}]
            )

    def extract_style_attributes(self, brief_embedding: torch.Tensor, 
                               image_features: List[torch.Tensor]) -> Dict[str, Any]:
        # Combine brief and image features
        combined_features = torch.cat([brief_embedding.unsqueeze(0)] + image_features)
        
        # Extract key style attributes
        style_attributes = {
            'silhouette': ['relaxed', 'tailored'],
            'materials': ['wool', 'cotton'],
            'colors': ['blue', 'grey'],
            'design_elements': ['pleats', 'pockets']
        }
        
        return style_attributes

    def build_style_graph(self, style_attributes: Dict[str, Any], 
                         brand_guidelines: Dict[str, Any]):
        g = Graph()
        ns = Namespace("http://fashion.org/")
        
        # Add style nodes
        for category, attributes in style_attributes.items():
            for attr in attributes:
                g.add((
                    URIRef(ns[category]),
                    URIRef(ns['has_attribute']),
                    Literal(attr)
                ))
                
        # Add brand guideline nodes
        for guideline, value in brand_guidelines.items():
            g.add((
                URIRef(ns['brand']),
                URIRef(ns[guideline]),
                Literal(str(value))
            ))
        
        # Save graph
        g.serialize("style_graph.ttl")

    def get_brand_context(self, style_attributes: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Query similar styles from brand history
        similar_styles = self.store.similarity_search(
            json.dumps(style_attributes),
            k=3
        )
        return [
            {'id': doc.metadata['id'], 'similarity': doc.metadata['score']}
            for doc in similar_styles
        ]

    def generate_design_direction(self, brief_embedding: torch.Tensor, 
                                style_attributes: Dict[str, Any]) -> Dict[str, Any]:
        # Generate concrete design direction
        direction = {
            'primary_elements': [
                {'type': 'silhouette', 'value': style_attributes['silhouette'][0]},
                {'type': 'material', 'value': style_attributes['materials'][0]}
            ],
            'secondary_elements': [
                {'type': 'color', 'value': style_attributes['colors'][0]},
                {'type': 'detail', 'value': style_attributes['design_elements'][0]}
            ],
            'constraints': [
                'maintain brand DNA',
                'seasonal appropriateness'
            ]
    }
        return direction

if __name__ == "__main__":
    main() 