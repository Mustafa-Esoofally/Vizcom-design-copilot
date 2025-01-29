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
        """Initialize the Design Analysis Agent with dual memory system."""
        # Initialize paths
        self.store_path = store_path
        
        # Initialize vision models
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize text models
        self.text_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Initialize memory systems
        self.long_term_memory = Chroma(
            collection_name="brand_guidelines",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(store_path, "long_term")
        )
        
        self.short_term_memory = Chroma(
            collection_name="project_context",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(store_path, "short_term")
        )
        
        # Initialize semantic graph
        self.style_graph = Graph()
        self.style_ns = Namespace("http://style.org/")
        self.brand_ns = Namespace("http://brand.org/")
        
    def analyze_brief(self, brief_text: str, reference_images: List[str], brand_guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze design brief and extract key style attributes."""
        # Extract text embeddings
        brief_embedding = self.text_model.encode(brief_text)
        
        # Process reference images
        image_features = []
        for img_path in reference_images:
            features = self.extract_image_features(img_path)
            image_features.append(features)
        
        # Store in short-term memory
        self.store_project_context(brief_text, image_features, brand_guidelines)
        
        # Extract style attributes
        style_attributes = self.extract_style_attributes(brief_embedding, image_features)
        
        # Build semantic relationships
        self.build_style_graph(style_attributes, brand_guidelines)
        
        return {
            'style_attributes': style_attributes,
            'brand_context': self.get_brand_context(style_attributes),
            'design_direction': self.generate_design_direction(brief_embedding, style_attributes)
        }
    
    def extract_image_features(self, image_path: str) -> torch.Tensor:
        """Extract visual features from reference image using CLIP."""
        image = self.clip_processor.image_from_path(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt")
        features = self.clip_model.get_image_features(**inputs)
        return features
    
    def store_project_context(self, brief_text: str, image_features: List[torch.Tensor], 
                            brand_guidelines: Dict[str, Any]):
        """Store project context in short-term memory."""
        # Create document for context
        context = {
            'brief': brief_text,
            'image_features': [f.tolist() for f in image_features],
            'brand_guidelines': brand_guidelines
        }
        
        self.short_term_memory.add_texts(
            texts=[brief_text],
            metadatas=[context]
        )
    
    def extract_style_attributes(self, brief_embedding: torch.Tensor, 
                               image_features: List[torch.Tensor]) -> Dict[str, Any]:
        """Extract style attributes from brief and reference images."""
        # Combine text and visual features
        combined_features = torch.cat([brief_embedding.unsqueeze(0)] + image_features)
        
        # Extract key style elements
        style_attributes = {
            'silhouette': self.analyze_silhouette(combined_features),
            'materials': self.analyze_materials(combined_features),
            'color_palette': self.analyze_colors(combined_features),
            'design_elements': self.analyze_design_elements(combined_features)
        }
        
        return style_attributes
    
    def build_style_graph(self, style_attributes: Dict[str, Any], 
                         brand_guidelines: Dict[str, Any]):
        """Build semantic graph of style relationships."""
        # Add style nodes
        for category, attributes in style_attributes.items():
            category_uri = URIRef(self.style_ns[category])
            for attr in attributes:
                self.style_graph.add((
                    category_uri,
                    self.style_ns.hasAttribute,
                    Literal(attr)
                ))
        
        # Add brand constraints
        for guideline, value in brand_guidelines.items():
            self.style_graph.add((
                URIRef(self.brand_ns[guideline]),
                self.brand_ns.constrains,
                Literal(value)
            ))
    
    def get_brand_context(self, style_attributes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant brand context from long-term memory."""
        # Create search query from style attributes
        query = " ".join([
            f"{category}: {', '.join(attrs)}"
            for category, attrs in style_attributes.items()
        ])
        
        # Search long-term memory
        results = self.long_term_memory.similarity_search(query)
        return [doc.metadata for doc in results]
    
    def generate_design_direction(self, brief_embedding: torch.Tensor, 
                                style_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level design direction based on analysis."""
        # Query project context
        context = self.short_term_memory.similarity_search(
            "design direction",
            k=3
        )
        
        # Combine with style attributes
        direction = {
            'primary_theme': self.extract_theme(brief_embedding),
            'key_elements': style_attributes['design_elements'][:3],
            'material_focus': style_attributes['materials'][:2],
            'color_strategy': self.generate_color_strategy(style_attributes['color_palette']),
            'silhouette_direction': style_attributes['silhouette'][:2]
        }
        
        return direction
    
    def analyze_silhouette(self, features: torch.Tensor) -> List[str]:
        """Analyze and extract silhouette attributes."""
        # Implement silhouette analysis logic
        return ['structured', 'relaxed', 'fitted']
    
    def analyze_materials(self, features: torch.Tensor) -> List[str]:
        """Analyze and extract material attributes."""
        # Implement material analysis logic
        return ['technical fabric', 'performance mesh', 'waterproof']
    
    def analyze_colors(self, features: torch.Tensor) -> List[str]:
        """Analyze and extract color attributes."""
        # Implement color analysis logic
        return ['navy', 'charcoal', 'red accent']
    
    def analyze_design_elements(self, features: torch.Tensor) -> List[str]:
        """Analyze and extract design element attributes."""
        # Implement design element analysis logic
        return ['asymmetric zip', 'reflective details', 'ventilation panels']
    
    def extract_theme(self, brief_embedding: torch.Tensor) -> str:
        """Extract primary theme from brief embedding."""
        # Implement theme extraction logic
        return "urban performance"
    
    def generate_color_strategy(self, colors: List[str]) -> Dict[str, List[str]]:
        """Generate color strategy from analyzed colors."""
        return {
            'primary': colors[:2],
            'accent': colors[2:],
            'combinations': [
                ['navy', 'red'],
                ['charcoal', 'red']
            ]
        }

if __name__ == "__main__":
    main() 