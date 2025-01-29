from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional
import logging
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import networkx as nx
from datetime import datetime
import json
import os
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageMemoryStore:
    """Memory store for fashion images with knowledge graph capabilities"""
    def __init__(self, cache_dir: Path = Path("Backend/cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.cache_dir / "fashion_memory.json"
        self.image_embeddings = {}
        self.knowledge_graph = nx.DiGraph()
        self.model = None
        self.processor = None
        self._load_memory()
    
    def _setup_clip(self):
        """Initialize CLIP model from Hugging Face"""
        try:
            model_name = "openai/clip-vit-large-patch14"
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            logger.info("CLIP model initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up CLIP model: {e}")
            raise
    
    def _load_memory(self):
        """Load stored memory from cache"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    # Convert stored lists back to numpy arrays
                    self.image_embeddings = {
                        k: np.array(v)
                        for k, v in data['embeddings'].items()
                    }
                    # Load graph data
                    graph_data = data['graph']
                    self.knowledge_graph = nx.node_link_graph(graph_data)
                logger.info("Loaded fashion memory from cache")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
                self._initialize_memory()
        else:
            self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize empty memory structures"""
        self.image_embeddings = {}
        self.knowledge_graph = nx.DiGraph()
        self._save_memory()
        logger.info("Initialized new fashion memory")
    
    def _save_memory(self):
        """Save memory to cache"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            embeddings_list = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.image_embeddings.items()
            }
            data = {
                'embeddings': embeddings_list,
                'graph': nx.node_link_data(self.knowledge_graph),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.memory_file, 'w') as f:
                json.dump(data, f)
            logger.info("Saved fashion memory to cache")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def analyze_seasons(self, season_dirs: List[Path]):
        """Analyze fashion images from multiple seasons"""
        if not self.model:
            self._setup_clip()
        
        total_images = sum(len(list(season_dir.glob("*.webp"))) for season_dir in season_dirs)
        processed = 0
        
        for season_dir in season_dirs:
            logger.info(f"Processing season: {season_dir.name}")
            for img_path in season_dir.glob("*.webp"):
                try:
                    self.analyze_image(img_path)
                    processed += 1
                    if processed % 10 == 0:
                        logger.info(f"Processed {processed}/{total_images} images")
                except Exception as e:
                    logger.error(f"Error processing {img_path.name}: {e}")
        
        self._save_memory()
        logger.info("Completed analyzing all seasons")
    
    def analyze_image(self, img_path: Path):
        """Analyze a single fashion image"""
        try:
            # Extract design elements from filename
            design_elements = self._extract_design_elements(img_path)
            
            # Get image embedding
            with Image.open(img_path) as image:
                # Convert RGBA to RGB if necessary
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                
                # Process image with CLIP
                inputs = self.processor(images=image, return_tensors="pt")
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features.detach().numpy()
                image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
                
                # Store embedding
                self.image_embeddings[str(img_path)] = image_features.squeeze(0)
            
            # Update knowledge graph
            self._update_knowledge_graph(design_elements, str(img_path))
            
        except Exception as e:
            logger.error(f"Error analyzing {img_path.name}: {e}")
            raise
    
    def _extract_design_elements(self, img_path: Path) -> Dict[str, str]:
        """Extract design elements from image filename"""
        name_parts = img_path.stem.split('-')
        elements = {
            'brand': name_parts[0] if len(name_parts) > 0 else '',
            'type': name_parts[1] if len(name_parts) > 1 else '',
            'color': name_parts[2] if len(name_parts) > 2 else '',
            'model': name_parts[3] if len(name_parts) > 3 else ''
        }
        return elements
    
    def _update_knowledge_graph(self, design_elements: Dict[str, str], img_path: str):
        """Update knowledge graph with design elements"""
        # Add nodes for each element
        for category, element in design_elements.items():
            if element:
                node_id = f"{category}:{element}"
                if not self.knowledge_graph.has_node(node_id):
                    self.knowledge_graph.add_node(node_id, 
                                               category=category,
                                               name=element)
                
                # Connect element to image
                self.knowledge_graph.add_edge(node_id, img_path, type='appears_in')
        
        # Add relationships between elements
        elements = list(design_elements.items())
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                cat1, elem1 = elements[i]
                cat2, elem2 = elements[j]
                if elem1 and elem2:
                    node1 = f"{cat1}:{elem1}"
                    node2 = f"{cat2}:{elem2}"
                    if self.knowledge_graph.has_edge(node1, node2):
                        self.knowledge_graph[node1][node2]['weight'] += 1
                    else:
                        self.knowledge_graph.add_edge(node1, node2, weight=1)
    
    def find_similar_designs(self, query_text: str, top_k: int = 5) -> List[str]:
        """Find similar designs based on text query"""
        if not self.model:
            raise ValueError("CLIP model not initialized. Please analyze seasons first.")
        
        try:
            # Encode query text
            inputs = self.processor(text=[query_text], return_tensors="pt", padding=True)
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features.detach().numpy()
            text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
            
            # Calculate similarities
            similarities = {}
            for img_path, img_embedding in self.image_embeddings.items():
                similarity = np.dot(text_features.squeeze(0), img_embedding)
                similarities[img_path] = float(similarity)
            
            # Sort by similarity
            similar_designs = sorted(similarities.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:top_k]
            
            return [path for path, _ in similar_designs]
        except Exception as e:
            logger.error(f"Error finding similar designs: {e}")
            return []
    
    def get_design_elements(self, img_path: str) -> Dict[str, List[str]]:
        """Get design elements for an image"""
        elements = defaultdict(list)
        try:
            for node in self.knowledge_graph.predecessors(img_path):
                category, element = node.split(':')
                elements[category].append(element)
        except Exception as e:
            logger.error(f"Error getting design elements for {img_path}: {e}")
        return dict(elements)
    
    def get_related_elements(self, category: str, element: str, 
                           min_weight: int = 2) -> List[str]:
        """Get related design elements"""
        node_id = f"{category}:{element}"
        if not self.knowledge_graph.has_node(node_id):
            return []
        
        try:
            related = []
            for _, target, data in self.knowledge_graph.edges(node_id, data=True):
                if data.get('weight', 0) >= min_weight:
                    cat, elem = target.split(':')
                    related.append(elem)
            return related
        except Exception as e:
            logger.error(f"Error getting related elements: {e}")
            return []

def main():
    """Main function to analyze fashion seasons"""
    try:
        # Set up directories
        base_dir = Path("Backend/designs")
        season_dirs = [base_dir / "Season_1", base_dir / "Season_2"]
        
        # Validate directories
        for season_dir in season_dirs:
            if not season_dir.exists():
                logger.error(f"Season directory not found: {season_dir}")
                return
        
        # Create memory store
        memory_store = ImageMemoryStore()
        
        # Analyze seasons
        print("\nAnalyzing fashion seasons...")
        memory_store.analyze_seasons(season_dirs)
        
        # Test similarity search
        test_queries = [
            "modern technical jacket",
            "classic denim pants",
            "elegant dress"
        ]
        
        print("\nTesting similarity search...")
        for query in test_queries:
            print(f"\nQuery: {query}")
            similar_designs = memory_store.find_similar_designs(query)
            
            print("Found similar designs:")
            for design_path in similar_designs:
                elements = memory_store.get_design_elements(design_path)
                print(f"\nDesign: {Path(design_path).name}")
                for category, items in elements.items():
                    print(f"{category}: {', '.join(items)}")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.exception("Detailed error information:")

if __name__ == "__main__":
    main() 