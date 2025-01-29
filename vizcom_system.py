from pathlib import Path
import torch
import logging
from typing import Dict, List, Optional, Tuple
import networkx as nx
from PIL import Image
import random
from fashion_clip.fashion_clip import FashionCLIP, FCLIPDataset
from diffusers import AutoPipelineForText2Image
import torch.nn.functional as F
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VizcomSystem:
    """Main system for fashion design analysis and generation"""
    def __init__(self, cache_dir: Path = Path("cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_graph = nx.DiGraph()
        
        # Initialize models
        try:
            # Initialize Fashion-CLIP
            self.setup_fashion_clip()
            
            # Initialize SDXL-Turbo for generation
            self.setup_generator()
            
            logger.info("Successfully initialized all models")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def setup_fashion_clip(self):
        """Initialize Fashion-CLIP model"""
        self.fclip = FashionCLIP('fashion-clip')
        if torch.cuda.is_available():
            self.fclip.model = self.fclip.model.to("cuda")
    
    def setup_generator(self):
        """Initialize SDXL-Turbo generator"""
        self.generator = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        if torch.cuda.is_available():
            self.generator.to("cuda")

    def process_input_designs(self, season_dirs: List[Path], text_brief: str):
        """Process input fashion designs and build knowledge graph"""
        # Create dataset from season directories
        catalog = []
        for season_dir in season_dirs:
            for img_path in season_dir.glob("*.webp"):
                catalog.append({
                    'id': str(img_path),
                    'image': str(img_path),
                    'caption': img_path.stem.replace('_', ' ')
                })
        
        self.dataset = FCLIPDataset('local_fashion',
                                  image_source_path=str(season_dirs[0].parent),
                                  image_source_type='local',
                                  catalog=catalog)
        
        # Extract design elements and build knowledge graph
        self._build_knowledge_graph(text_brief)
        
        logger.info(f"Processed {len(catalog)} designs and built knowledge graph")

    def _build_knowledge_graph(self, text_brief: str):
        """Build knowledge graph from design elements"""
        # Get embeddings for the text brief
        brief_embedding = self.fclip.encode_text([text_brief], batch_size=1)[0]
        brief_embedding = brief_embedding / np.linalg.norm(brief_embedding, ord=2)
        
        # Get all image paths
        image_paths = [str(item['image']) for item in self.dataset.catalog]
        
        # Get image embeddings in batches
        image_embeddings = self.fclip.encode_images(image_paths, batch_size=32)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
        
        # Process each design and add to graph
        for idx, item in enumerate(self.dataset.catalog):
            img_path = Path(item['image'])
            name_parts = img_path.stem.split('-')  # Changed from split('_') to split('-')
            
            # Calculate similarity using dot product (since vectors are normalized)
            similarity = brief_embedding.dot(image_embeddings[idx])
            
            # Add nodes and edges
            self._add_design_to_graph(name_parts, float(similarity))
            
        logger.info(f"Built knowledge graph with {len(self.knowledge_graph.nodes)} nodes")

    def _add_design_to_graph(self, name_parts: List[str], similarity: float):
        """Add design elements and relationships to knowledge graph"""
        # Skip if not enough parts or doesn't start with 'entire-studios'
        if len(name_parts) < 4 or name_parts[0] != 'entire' or name_parts[1] != 'studios':
            return
            
        # Extract meaningful parts from the name
        # Format: entire-studios-[item-name]-[color]-[model]-[number]
        # Example: entire-studios-father-suiting-pant-ash-tricky-01
        
        # Remove 'entire-studios' prefix and model-number suffix
        design_parts = name_parts[2:-2]  # Skip prefix and model-number
        
        # Handle multi-word items (e.g., 'double-breasted-blazer')
        item_parts = []
        color_found = False
        for part in design_parts:
            # Common colors in our dataset
            colors = {'ash', 'grey', 'stone', 'light', 'elephant', 'cinder', 'ashen', 'hazelnut', 
                     'grove', 'hunter', 'moth', 'seal', 'rutile', 'taupe', 'frost'}
            
            if part in colors:
                color_found = True
                color = part
                break
            item_parts.append(part)
        
        # Map parts to categories
        categories = {
            'base_type': item_parts[0],  # e.g., 'father' or 'double'
            'style': item_parts[1] if len(item_parts) > 1 else None,  # e.g., 'suiting' or 'breasted'
            'category': item_parts[-1] if len(item_parts) > 1 else None,  # e.g., 'pant' or 'blazer'
            'color': color if color_found else None
        }
        
        nodes = []
        
        # Add nodes for each valid design element
        for category, element in categories.items():
            if element:
                node_id = f"{category}:{element}"
                self.knowledge_graph.add_node(
                    node_id,
                    category=category,
                    name=element,
                    relevance=similarity
                )
                nodes.append(node_id)
        
        # Add edges between elements with weighted relationships
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                weight = similarity * (1.0 - (j - i) * 0.1)  # Decrease weight with distance
                self.knowledge_graph.add_edge(nodes[i], nodes[j], weight=weight)
                
        # Add special edges for common combinations
        if 'base_type:father' in nodes and 'category:pant' in nodes:
            self.knowledge_graph.add_edge('base_type:father', 'category:pant', weight=similarity * 1.2)
        if 'style:suiting' in nodes and 'category:pant' in nodes:
            self.knowledge_graph.add_edge('style:suiting', 'category:pant', weight=similarity * 1.2)

    def generate_design_paths(self, num_paths: int = 5) -> List[Dict]:
        """Generate design paths based on knowledge graph"""
        paths = []
        nodes = list(self.knowledge_graph.nodes(data=True))
        nodes.sort(key=lambda x: x[1].get('relevance', 0), reverse=True)
        
        for _ in range(num_paths):
            path = self._generate_single_path(nodes)
            if path:
                paths.append(path)
                
        return paths

    def _generate_single_path(self, nodes) -> Optional[Dict]:
        """Generate a single design path"""
        try:
            # Start with high-relevance base type
            base_node = next((n for n in nodes if n[1]['category'] == 'base_type'), None)
            if not base_node:
                return None
                
            path = {
                'base_type': base_node[1]['name'],
                'elements': [],
                'description': []
            }
            
            # Add related elements based on edge weights
            current = base_node[0]
            categories_found = {'base_type'}
            
            # Try to find elements in this order
            desired_categories = ['style', 'category', 'color']
            
            for category in desired_categories:
                neighbors = [(n, self.knowledge_graph[current][n]['weight']) 
                           for n in self.knowledge_graph.neighbors(current)
                           if n.split(':')[0] == category]
                if neighbors:
                    next_node = max(neighbors, key=lambda x: x[1])[0]
                    element = self.knowledge_graph.nodes[next_node]['name']
                    path['elements'].append(element)
                    categories_found.add(category)
                    current = next_node
            
            # Build natural description
            desc_parts = []
            if 'base_type' in categories_found:
                desc_parts.append(path['base_type'])
            if 'style' in categories_found:
                desc_parts.append(path['elements'][0])
            if 'category' in categories_found:
                desc_parts.append(path['elements'][1])
            if 'color' in categories_found:
                desc_parts.append(f"in {path['elements'][2]}")
                
            path['description'] = ' '.join(desc_parts)
            return path
            
        except Exception as e:
            logger.error(f"Error generating path: {e}")
            return None

    def generate_designs(self, selected_path: Dict, num_variations: int = 3) -> List[Image.Image]:
        """Generate design variations based on selected path"""
        images = []
        
        try:
            # Prepare enhanced prompt using fashion-specific language
            base_prompt = f"a {selected_path['description']}"
            
            # Add style modifiers from FashionCLIP paper
            style_prompt = (
                f"{base_prompt}, high-end fashion photography, professional studio lighting, "
                f"clean background, detailed fabric texture, photorealistic, high quality, "
                f"fashion catalog style, front view, centered composition"
            )
            
            negative_prompt = (
                "low quality, blurry, distorted, unrealistic, amateur, text, watermark, "
                "deformed, disfigured, bad proportions, duplicate, multiple views, "
                "collage, side view, cropped, out of frame"
            )
            
            # Generate variations
            for _ in range(num_variations):
                output = self.generator(
                    prompt=style_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                ).images[0]
                images.append(output)
                
            logger.info(f"Generated {len(images)} design variations")
            
        except Exception as e:
            logger.error(f"Error generating designs: {e}")
            
        return images 