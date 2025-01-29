import torch
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json
import networkx as nx
import numpy as np
from datetime import datetime
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoPipelineForText2Image
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FashionKnowledgeSystem:
    """Fashion design knowledge system using Fashion-CLIP for visual-language understanding"""
    
    def __init__(self, cache_dir: Path = Path("Backend/cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.cache_dir / "fashion_knowledge.json"
        self.embeddings_file = self.cache_dir / "fashion_embeddings.pt"
        self.knowledge_graph = nx.DiGraph()
        
        # Initialize Fashion-CLIP
        try:
            self.clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
            self.clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            
            # Initialize SDXL-Turbo for generation
            self.image_generator = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16"
            )
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")
                self.image_generator.to("cuda")
            
            logger.info("Successfully initialized Fashion-CLIP and SDXL-Turbo models")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
        
        self.image_embeddings = {}
        self.text_embeddings = {}
        self._load_memory()
    
    def _load_memory(self):
        """Load stored memory and embeddings"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    graph_data = data['graph']
                    self.knowledge_graph = nx.node_link_graph(graph_data)
                    
                # Load embeddings if they exist
                if self.embeddings_file.exists():
                    embeddings = torch.load(self.embeddings_file)
                    self.image_embeddings = embeddings.get('image_embeddings', {})
                    self.text_embeddings = embeddings.get('text_embeddings', {})
                    
                logger.info("Loaded fashion knowledge from cache")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
                self._initialize_memory()
        else:
            self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize empty memory structures"""
        self.knowledge_graph = nx.DiGraph()
        self.image_embeddings = {}
        self.text_embeddings = {}
        self._save_memory()
        logger.info("Initialized new fashion knowledge")
    
    def _save_memory(self):
        """Save memory to cache"""
        try:
            # Save graph structure
            data = {
                'graph': nx.node_link_data(self.knowledge_graph),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.memory_file, 'w') as f:
                json.dump(data, f)
            
            # Save embeddings
            torch.save({
                'image_embeddings': self.image_embeddings,
                'text_embeddings': self.text_embeddings
            }, self.embeddings_file)
            
            logger.info("Saved fashion knowledge to cache")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def add_design_elements(self, elements: Dict[str, str], image_path: str):
        """Add design elements and compute embeddings using Fashion-CLIP"""
        try:
            # Process image
            with Image.open(image_path) as image:
                inputs = self.clip_processor(images=image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                self.image_embeddings[image_path] = image_features
            
            # Add nodes for each element
            for category, element in elements.items():
                if element:
                    node_id = f"{category}:{element}"
                    if not self.knowledge_graph.has_node(node_id):
                        self.knowledge_graph.add_node(node_id,
                                                   category=category,
                                                   name=element)
                        
                        # Compute text embedding using Fashion-CLIP
                        text = f"{element} {category}"
                        text_inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
                        if torch.cuda.is_available():
                            text_inputs = {k: v.to("cuda") for k, v in text_inputs.items()}
                        
                        with torch.no_grad():
                            text_features = self.clip_model.get_text_features(**text_inputs)
                        self.text_embeddings[node_id] = text_features
                    
                    # Connect element to image
                    self.knowledge_graph.add_edge(node_id, image_path, type='appears_in')
            
            # Add relationships between elements
            elements_list = list(elements.items())
            for i in range(len(elements_list)):
                for j in range(i + 1, len(elements_list)):
                    cat1, elem1 = elements_list[i]
                    cat2, elem2 = elements_list[j]
                    if elem1 and elem2:
                        node1 = f"{cat1}:{elem1}"
                        node2 = f"{cat2}:{elem2}"
                        
                        # Calculate similarity using Fashion-CLIP embeddings
                        if node1 in self.text_embeddings and node2 in self.text_embeddings:
                            similarity = F.cosine_similarity(
                                self.text_embeddings[node1],
                                self.text_embeddings[node2]
                            ).item()
                            
                            # Add weighted edge based on similarity
                            self.knowledge_graph.add_edge(node1, node2, weight=similarity)
            
            self._save_memory()
            
        except Exception as e:
            logger.error(f"Error adding design elements: {e}")
            raise
    
    def get_element_suggestions(self, element_type: str, element_name: str, 
                              top_k: int = 5) -> List[Tuple[str, float]]:
        """Get suggestions using Fashion-CLIP embeddings"""
        try:
            node_id = f"{element_type}:{element_name}"
            if node_id not in self.text_embeddings:
                return []
            
            query_embedding = self.text_embeddings[node_id]
            suggestions = []
            
            # Compare with all other elements using cosine similarity
            for other_id, other_embedding in self.text_embeddings.items():
                if other_id != node_id:
                    similarity = F.cosine_similarity(
                        query_embedding,
                        other_embedding
                    ).item()
                    
                    category = self.knowledge_graph.nodes[other_id].get('category')
                    name = self.knowledge_graph.nodes[other_id].get('name')
                    if category and name:
                        suggestions.append((f"{name} ({category})", similarity))
            
            # Sort by similarity and get top-k
            suggestions.sort(key=lambda x: x[1], reverse=True)
            return suggestions[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting element suggestions: {e}")
            return []
    
    def generate_design_image(self, prompt: str) -> Optional[Image.Image]:
        """Generate a new design image using SDXL-Turbo"""
        try:
            # Add fashion-specific context to the prompt
            fashion_prompt = f"A high-quality fashion photograph of {prompt}, professional studio lighting, high-end fashion magazine style"
            
            # Generate image
            image = self.image_generator(
                prompt=fashion_prompt,
                num_inference_steps=1,  # Turbo is optimized for speed
                guidance_scale=0.0,     # No guidance needed for Turbo
            ).images[0]
            
            return image
            
        except Exception as e:
            logger.error(f"Error generating design image: {e}")
            return None
    
    def get_design_path(self, start_element: str, end_element: str) -> List[str]:
        """Find path between design elements using similarity-weighted edges"""
        try:
            if not (self.knowledge_graph.has_node(start_element) and 
                   self.knowledge_graph.has_node(end_element)):
                return []
            
            # Use similarity weights for path finding
            path = nx.shortest_path(self.knowledge_graph, 
                                  start_element, 
                                  end_element, 
                                  weight='weight')
            return path
            
        except Exception as e:
            logger.error(f"Error finding design path: {e}")
            return []
    
    def generate_prompt_from_path(self, path: List[str]) -> str:
        """Generate a detailed prompt from a design path"""
        try:
            elements = []
            for node in path:
                if ':' in node:
                    category, element = node.split(':')
                    elements.append(f"{element} {category}")
            
            # Create a more descriptive prompt
            prompt = "Create a fashion design that combines " + ", ".join(elements[:-1])
            if elements:
                prompt += f" with {elements[-1]}"
            prompt += ", ensuring a cohesive and stylish look"
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error generating prompt from path: {e}")
            return "" 