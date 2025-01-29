import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as patches
import re
import torch
from PIL import Image
import random
import importlib.util
import os
import sys
from fashion_clip.fashion_clip import FashionCLIP, FCLIPDataset
from sgm.util import instantiate_from_config
from omegaconf import OmegaConf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DesignKnowledgeGraph:
    """Knowledge graph for design elements with interactive visualization"""
    def __init__(self, season_dirs: List[Path]):
        self.G = nx.DiGraph()
        self.season_dirs = season_dirs
        self._setup_fashion_clip()
        self._extract_categories_from_data()
        self._build_knowledge_graph()
        self.category_order = ["Base", "Material", "Style", "Color", "Fit", "Details"]
    
    def _setup_fashion_clip(self):
        """Initialize Fashion CLIP for fashion understanding"""
        catalog = []
        for season_dir in self.season_dirs:
            for img_path in Path(season_dir).glob("*.webp"):
                catalog.append({
                    'id': str(img_path),
                    'image': str(img_path),
                    'caption': img_path.stem.replace('_', ' ')
                })
        
        self.dataset = FCLIPDataset('local_fashion',
                                  image_source_path=str(Path(self.season_dirs[0]).parent),
                                  image_source_type='local',
                                  catalog=catalog)
        self.fclip = FashionCLIP('fashion-clip', self.dataset)
    
    def _extract_categories_from_data(self):
        """Extract categories and elements from fashion data"""
        self.categories = {
            'base_type': set(),
            'material': set(),
            'style': set(),
            'color': set(),
            'fit': set(),
            'details': set()
        }
        
        for season_dir in self.season_dirs:
            for img_path in Path(season_dir).glob("*.webp"):
                name_parts = img_path.stem.split('_')
                if len(name_parts) >= 3:
                    self.categories['base_type'].add(name_parts[0])
                    self.categories['material'].add(name_parts[1])
                    self.categories['style'].add(name_parts[2])
                    if len(name_parts) > 3:
                        self.categories['color'].add(name_parts[3])
                    if len(name_parts) > 4:
                        self.categories['fit'].add(name_parts[4])
                    if len(name_parts) > 5:
                        self.categories['details'].update(name_parts[5:])
    
    def _build_knowledge_graph(self):
        """Build the knowledge graph with nodes and relationships"""
        # Add category nodes
        category_colors = {
            'base_type': '#FF6B6B',  # Red
            'material': '#4ECDC4',   # Teal
            'style': '#45B7D1',      # Blue
            'color': '#96CEB4',      # Green
            'fit': '#FFEEAD',        # Yellow
            'details': '#FF9999'     # Light Red
        }
        
        # Add nodes for each category and their elements
        for category, elements in self.categories.items():
            for element in elements:
                node_id = f"{category}:{element}"
                self.G.add_node(node_id, 
                              category=category,
                              name=element,
                              color=category_colors[category])
        
        # Add relationships between elements based on co-occurrence in designs
        for season_dir in self.season_dirs:
            for img_path in Path(season_dir).glob("*.webp"):
                name_parts = img_path.stem.split('_')
                if len(name_parts) >= 3:
                    # Create edges between consecutive elements
                    nodes = []
                    if name_parts[0]:
                        nodes.append(f"base_type:{name_parts[0]}")
                    if name_parts[1]:
                        nodes.append(f"material:{name_parts[1]}")
                    if name_parts[2]:
                        nodes.append(f"style:{name_parts[2]}")
                    if len(name_parts) > 3 and name_parts[3]:
                        nodes.append(f"color:{name_parts[3]}")
                    if len(name_parts) > 4 and name_parts[4]:
                        nodes.append(f"fit:{name_parts[4]}")
                    if len(name_parts) > 5:
                        for detail in name_parts[5:]:
                            if detail:
                                nodes.append(f"details:{detail}")
                    
                    # Add edges with weights
                    for i in range(len(nodes)-1):
                        for j in range(i+1, len(nodes)):
                            if self.G.has_edge(nodes[i], nodes[j]):
                                self.G[nodes[i]][nodes[j]]['weight'] += 1
                            else:
                                self.G.add_edge(nodes[i], nodes[j], weight=1)
    
    def generate_paths(self, base_prompt):
        """Generate design paths based on fashion understanding"""
        # Use Fashion CLIP to understand the prompt
        similar_items = self.fclip.retrieval([base_prompt])
        
        paths = []
        for item_idx in similar_items[0][:5]:  # Get top 5 similar items
            item = self.dataset.catalog[item_idx]
            name_parts = Path(item['image']).stem.split('_')
            
            # Get related elements from knowledge graph
            related_elements = self._get_related_elements(name_parts)
            
            path = {
                'base_type': name_parts[0],
                'material': name_parts[1],
                'style': name_parts[2],
                'elements': related_elements,
                'description': f"A {' '.join(name_parts)} design inspired by similar items in our fashion database"
            }
            paths.append(path)
            
        return paths
    
    def _get_related_elements(self, name_parts):
        """Get related design elements from the knowledge graph"""
        related_elements = []
        if len(name_parts) < 3:
            return related_elements
            
        base_node = f"base_type:{name_parts[0]}"
        if base_node in self.G:
            # Get most strongly connected nodes
            neighbors = [(n, self.G[base_node][n]['weight']) 
                        for n in self.G.neighbors(base_node)]
            neighbors.sort(key=lambda x: x[1], reverse=True)
            
            # Add top related elements
            for node, _ in neighbors[:3]:
                category, element = node.split(':')
                if element not in name_parts:
                    related_elements.append(element)
        
        return related_elements
    
    def generate_designs(self, selected_path, num_variations=3):
        """Generate design variations using generative models"""
        try:
            # Initialize the generator
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = initialize_generator(device)
            if generator is None:
                raise Exception("Failed to initialize image generator")
            
            # Prepare enhanced prompt
            elements = selected_path['elements']
            base_prompt = f"A {selected_path['style']} {selected_path['base_type']} made of {selected_path['material']}"
            if elements:
                base_prompt += f" with {', '.join(elements)}"
            
            # Add style modifiers for better generation
            style_prompt = (
                f"{base_prompt}, high-end fashion photography, professional studio lighting, "
                f"clean background, detailed fabric texture, photorealistic, high quality"
            )
            
            negative_prompt = (
                "low quality, blurry, distorted, unrealistic, amateur, text, watermark, "
                "deformed, disfigured, bad proportions"
            )
            
            # Generate variations
            images = []
            for _ in range(num_variations):
                output = generator(
                    prompt=style_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5
                ).images[0]
                images.append(output)
            
            # Save generated images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path("results") / timestamp
            save_dir.mkdir(parents=True, exist_ok=True)
            
            image_paths = []
            for i, image in enumerate(images):
                save_path = save_dir / f"variation_{i+1}.png"
                image.save(save_path)
                image_paths.append(save_path)
                
            return image_paths
            
        except Exception as e:
            logger.error(f"Error during design generation: {str(e)}")
            return []
    
    def visualize(self):
        """Visualize the knowledge graph"""
        plt.figure(figsize=(15, 10))
        
        # Use spring layout with adjusted parameters for better visualization
        pos = nx.spring_layout(self.G, k=2, iterations=50)
        
        # Draw nodes with category-based colors
        for category in self.categories:
            nodes = [n for n in self.G.nodes() if n.startswith(f"{category}:")]
            if nodes:
                nx.draw_networkx_nodes(
                    self.G, pos,
                    nodelist=nodes,
                    node_color=[self.G.nodes[n]['color'] for n in nodes],
                    node_size=1500,
                    alpha=0.7
                )
        
        # Draw edges with weights influencing width
        edge_weights = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [w/max_weight * 2 for w in edge_weights]
            nx.draw_networkx_edges(self.G, pos, width=edge_widths, alpha=0.5)
        
        # Add labels
        labels = {node: self.G.nodes[node]['name'] for node in self.G.nodes()}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=8)
        
        plt.title("Fashion Design Knowledge Graph")
        plt.axis('off')
        
        # Save with high DPI for better quality
        save_dir = Path("results")
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / "knowledge_graph.png", dpi=300, bbox_inches='tight')
        plt.close()

def save_design_path(base_prompt: str, selections: Dict[str, str], results_dir: Path) -> Path:
    """Save the selected design path to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_file = results_dir / f"design_path_{timestamp}.txt"
    
    with open(path_file, "w") as f:
        f.write(f"Base Prompt: {base_prompt}\n\n")
        f.write("Selected Path:\n")
        f.write("=============\n")
        for category in ["Base", "Material", "Style", "Color", "Fit", "Details"]:
            if category in selections:
                f.write(f"{category}: {selections[category]}\n")
        
        # Write the generation prompt that would be used
        f.write("\nGeneration Prompt:\n")
        f.write("=================\n")
        details = " ".join(f"{v.lower()}" for k, v in selections.items() if k != "Base")
        prompt = f"{base_prompt}, {details}, high-end fashion photography, studio lighting, clean background, detailed fabric texture, photorealistic"
        f.write(prompt)
    
    return path_file

def find_reference_designs(season_dirs: List[Path], selections: Dict[str, str]) -> List[Path]:
    """Find relevant reference designs from the seasonal collections"""
    reference_designs = []
    
    # Convert selections to lowercase for matching
    selections_lower = {k: v.lower() for k, v in selections.items()}
    
    for season_dir in season_dirs:
        for file_path in season_dir.glob("*.webp"):
            filename = file_path.stem.lower()
            
            # Check if file matches our selected attributes
            matches = 0
            for category, value in selections_lower.items():
                if value in filename:
                    matches += 1
            
            # If file matches at least 2 attributes, include it as reference
            if matches >= 2:
                reference_designs.append(file_path)
    
    return reference_designs

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = {
        'diffusers': 'diffusers',
        'transformers': 'transformers',
        'torch': 'torch'
    }
    
    missing_packages = []
    for package, import_name in required_packages.items():
        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(package)
    
    return missing_packages

def initialize_generator(device="cpu"):
    """Initialize the image generation model with proper error handling"""
    try:
        from diffusers import StableDiffusionPipeline
        
        # Try loading the model with minimal components
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        if device == "cuda" and torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            logger.warning("CUDA not available, using CPU for generation (this will be slow)")
        
        return pipe
    
    except Exception as e:
        logger.error(f"Error initializing generator: {e}")
        return None

def generate_designs(base_prompt: str, selections: Dict[str, str], 
                    reference_designs: List[Path], results_dir: Path,
                    num_designs: int = 3) -> List[Tuple[Path, Path]]:
    """Generate multiple designs based on selections and references"""
    generated_files = []
    
    # Check dependencies first
    missing_packages = check_dependencies()
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages and try again")
        return generated_files
    
    # Initialize generator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = initialize_generator(device)
    
    if generator is None:
        logger.error("Failed to initialize image generator")
        return generated_files
    
    # Construct base attributes string
    details = " ".join(f"{v.lower()}" for k, v in selections.items() if k != "Base")
    
    # Get reference design descriptions
    reference_descriptions = []
    for ref_path in reference_designs:
        filename = ref_path.stem
        desc = filename.replace("-", " ").replace("entire studios", "").strip()
        reference_descriptions.append(desc)
    
    for i in range(num_designs):
        try:
            # Select a random reference for inspiration
            ref_desc = random.choice(reference_descriptions) if reference_descriptions else ""
            
            # Create variation in the prompt
            style_variations = [
                "contemporary fashion design",
                "high-end designer clothing",
                "luxury fashion piece"
            ]
            detail_variations = [
                "with clean lines and modern details",
                "featuring minimalist design elements",
                "with sophisticated construction"
            ]
            
            # Construct prompt with variations
            prompt = (
                f"professional fashion photography of a {details} {selections['Base'].lower()}, "
                f"inspired by {ref_desc}, {random.choice(style_variations)}, "
                f"{random.choice(detail_variations)}, "
                f"studio lighting, clean background, detailed fabric texture, photorealistic"
            )
            
            negative_prompt = (
                "low quality, blurry, distorted, unrealistic, amateur, text, watermark, "
                "deformed, disfigured, extra limbs, bad proportions"
            )
            
            # Generate image with variations in parameters
            with torch.no_grad():
                try:
                    output = generator(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=20,  # Reduced steps for faster generation
                        guidance_scale=7.5,
                        height=512,  # Reduced size for faster generation
                        width=512
                    )
                    image = output.images[0]
                except Exception as e:
                    logger.error(f"Error during image generation: {e}")
                    continue
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = results_dir / f"generated_design_{i+1}_{timestamp}.png"
            prompt_path = results_dir / f"prompt_{i+1}_{timestamp}.txt"
            
            # Save image and prompt
            image.save(image_path)
            with open(prompt_path, "w") as f:
                f.write(f"Base Prompt: {base_prompt}\n\n")
                f.write("Selected Attributes:\n")
                f.write("===================\n")
                for category, value in selections.items():
                    f.write(f"{category}: {value}\n")
                f.write("\nReference Design:\n")
                f.write(f"{ref_desc}\n\n")
                f.write("Generation Prompt:\n")
                f.write("=================\n")
                f.write(prompt)
                f.write("\n\nNegative Prompt:\n")
                f.write("===============\n")
                f.write(negative_prompt)
            
            generated_files.append((image_path, prompt_path))
            logger.info(f"Generated design {i+1}/3")
            
        except Exception as e:
            logger.error(f"Error generating design {i+1}: {e}")
            continue
    
    return generated_files

def main():
    """Main function to process designs and generate paths"""
    try:
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Set up season directories
        base_dir = Path("Backend/designs")
        season_dirs = [base_dir / "Season_1", base_dir / "Season_2"]
        
        # Validate season directories
        for season_dir in season_dirs:
            if not season_dir.exists():
                logger.error(f"Season directory not found: {season_dir}")
                return
        
        # Create knowledge graph from fashion data
        print("\nAnalyzing fashion data from seasonal collections...")
        graph = DesignKnowledgeGraph(season_dirs)
        
        # Generate and save knowledge graph visualization
        print("Generating knowledge graph visualization...")
        graph.visualize()
        print("Knowledge graph saved to results/knowledge_graph.png")
        
        # Get base prompt from user
        print("\nWelcome to the Design Path Generator!")
        print("=================================")
        base_prompt = input("Enter your base design prompt (e.g., 'technical pant'): ").strip()
        
        if not base_prompt:
            print("Please provide a valid design prompt.")
            return
        
        # Find matching paths
        print("\nAnalyzing design paths...")
        matching_paths = graph.generate_paths(base_prompt)
        
        if not matching_paths:
            print("\nNo matching design paths found. Please try a different prompt.")
            return
        
        # Display matching paths
        print("\nFound these design paths that match your prompt:")
        print("=============================================")
        for i, path in enumerate(matching_paths, 1):
            print(f"\nPath {i}:")
            print("--------")
            print(f"Base Type: {path['base_type']}")
            print(f"Material: {path['material']}")
            print(f"Style: {path['style']}")
            if path['elements']:
                print(f"Additional Elements: {', '.join(path['elements'])}")
            print(f"Description: {path['description']}")
        
        # Get user selection
        while True:
            try:
                choice = input("\nSelect a path (1-5) to generate designs: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(matching_paths):
                    selected_path = matching_paths[idx]
                    break
                else:
                    print("Invalid selection. Please enter a number between 1 and", len(matching_paths))
            except ValueError:
                print("Please enter a valid number.")
        
        # Generate designs
        print("\nGenerating design variations...")
        try:
            generated_files = graph.generate_designs(selected_path)
            
            if generated_files:
                print("\nGeneration complete!")
                print("===================")
                for i, image_path in enumerate(generated_files, 1):
                    print(f"\nDesign {i}:")
                    print(f"Image: {image_path}")
                
                # Save the design path for reference
                path_file = save_design_path(base_prompt, selected_path, results_dir)
                print(f"\nDesign path saved to: {path_file}")
            else:
                print("\nNo designs were generated. Please try again with a different prompt or path.")
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            path_file = save_design_path(base_prompt, selected_path, results_dir)
            print(f"\nDesign path saved to: {path_file}")
            print("You can use this path information to generate images later")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print("Please try again. If the problem persists, check the logs for more information.")

if __name__ == "__main__":
    main() 