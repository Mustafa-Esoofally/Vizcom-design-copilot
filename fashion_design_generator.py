import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
from PIL import Image
import torch
import re
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
from fashion_knowledge_graph import extract_fashion_concepts
import networkx as nx
from collections import defaultdict

# Load environment variables
load_dotenv()

def load_knowledge_graph() -> Dict:
    """Load the fashion knowledge graph data."""
    with open("visualizations/fashion_graph_data.json", 'r') as f:
        return json.load(f)

def get_node_fashion_elements(node: str) -> Dict[str, Set[str]]:
    """Extract fashion elements from a node's description."""
    elements = defaultdict(set)
    desc_path = f"descriptions/{node}.txt"
    if os.path.exists(desc_path):
        with open(desc_path, 'r', encoding='utf-8') as f:
            desc = f.read()
            concepts = extract_fashion_concepts(desc)
            for category, items in concepts.items():
                elements[category].update(items)
    return elements

def find_relevant_paths(G: nx.Graph, input_text: str, max_paths: int = 5) -> List[List[str]]:
    """Find relevant paths based on fashion elements and input text."""
    keywords = input_text.lower().split()
    
    # Create index of nodes by fashion elements
    element_to_nodes = defaultdict(set)
    style_to_nodes = defaultdict(set)
    node_elements = {}
    
    # Define common fashion elements
    material_keywords = {'cotton', 'silk', 'wool', 'leather', 'denim', 'jersey', 'knit', 'fleece', 'canvas'}
    color_keywords = {'black', 'white', 'red', 'blue', 'green', 'grey', 'taupe', 'sage', 'olive', 'beige', 'cream'}
    
    # Index all nodes by their elements
    for node in G.nodes():
        elements = get_node_fashion_elements(node)
        node_elements[node] = elements
        node_type = G.nodes[node]['type'].lower()
        
        # Add node type as an element
        elements['type'] = {node_type}
        
        # Index nodes by their elements
        for category, items in elements.items():
            for item in items:
                if item.lower() in material_keywords:
                    element_to_nodes[f"material_{item.lower()}"].add(node)
                elif item.lower() in color_keywords:
                    element_to_nodes[f"color_{item.lower()}"].add(node)
                elif category == 'styles':
                    style_to_nodes[item].add(node)
                else:
                    element_to_nodes[item].add(node)
    
    # Find relevant starting points
    start_nodes = set()
    for node in G.nodes():
        node_type = G.nodes[node]['type'].lower()
        elements = node_elements[node]
        
        # Match by type (pant) or color (blue)
        if any(keyword in node_type for keyword in keywords) or \
           any(keyword in ' '.join(str(item) for items in elements.values() for item in items).lower() for keyword in keywords):
            start_nodes.add(node)
    
    # Get available materials and colors
    materials = sorted(k for k in element_to_nodes.keys() if k.startswith('material_'))
    colors = sorted(k for k in element_to_nodes.keys() if k.startswith('color_'))
    styles = sorted(style_to_nodes.keys())
    
    # Create diverse paths
    paths = []
    seen_combinations = set()
    
    for start_node in start_nodes:
        # Try different combinations of materials, colors, and styles
        for material in materials:
            material_nodes = element_to_nodes[material] - {start_node}
            if not material_nodes:
                continue
                
            for color in colors:
                color_nodes = element_to_nodes[color] - {start_node} - material_nodes
                if not color_nodes:
                    continue
                    
                for style in styles:
                    # Create unique combination
                    combination = (material, color, style)
                    if combination in seen_combinations:
                        continue
                    
                    style_nodes = style_to_nodes[style] - {start_node} - material_nodes - color_nodes
                    if not style_nodes:
                        continue
                    
                    # Create path with unique elements
                    path = [
                        start_node,
                        min(material_nodes),  # Material-focused design
                        min(color_nodes),     # Color-focused design
                        min(style_nodes)      # Style-focused design
                    ]
                    
                    if len(set(path)) == len(path):  # Ensure all nodes are unique
                        paths.append(path)
                        seen_combinations.add(combination)
                        
                        if len(paths) >= max_paths:
                            return paths
    
    return paths[:max_paths]

def present_path_options(G: nx.Graph, paths: List[List[str]]) -> Dict:
    """Present path options to the user and return the selected path's design elements."""
    print("\nFound the following design paths:")
    print("-" * 80)
    
    path_elements = []
    for i, path in enumerate(paths):
        # Get design elements from each node in the path
        elements = defaultdict(set)
        for node in path:
            node_elements = get_node_fashion_elements(node)
            for category, items in node_elements.items():
                elements[category].update(items)
        
        # Format path information
        path_info = {
            'nodes': path,
            'elements': {k: list(v) for k, v in elements.items() if k != 'styles'},
            'styles': list(elements.get('styles', set()))
        }
        path_elements.append(path_info)
        
        # Display path information
        print(f"\nOption {i+1}:")
        print(f"Design Path: {' -> '.join(G.nodes[node]['type'] for node in path)}")
        print("Elements by category:")
        for category, items in elements.items():
            if category != 'styles':
                print(f"  {category.title()}: {', '.join(items)}")
        print(f"Styles: {', '.join(elements.get('styles', []))}")
        print("-" * 80)
    
    # Get user selection
    while True:
        try:
            choice = int(input("\nSelect a design path (1-{}): ".format(len(paths))))
            if 1 <= choice <= len(paths):
                return path_elements[choice-1]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

def generate_prompt_from_path(input_text: str, path_info: Dict) -> str:
    """Generate a prompt using the selected path's design elements."""
    prompt = f"High-end fashion design: {input_text}"
    
    # Add elements by category
    for category, items in path_info['elements'].items():
        if items:
            # Select most relevant items (up to 2 per category)
            key_items = items[:2]
            prompt += f", with {category} {', '.join(key_items)}"
    
    if path_info['styles']:
        # Select most relevant styles (up to 2)
        key_styles = path_info['styles'][:2]
        prompt += f", in a {', '.join(key_styles)} style"
    
    prompt += ". Professional fashion photography"
    return prompt

def generate_designs(prompt: str, num_images: int = 5) -> List[Image.Image]:
    """Generate new fashion designs using Stable Diffusion."""
    # Initialize Stable Diffusion pipeline with fashion-specific model
    model_id = "runwayml/stable-diffusion-v1-5"  # Using a more recent model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_auth_token=os.getenv("HUGGING_FACE_TOKEN")
    ).to("cuda")
    
    # Set up generation parameters
    generator = torch.Generator(device="cuda").manual_seed(42)  # For reproducibility
    
    # Generate images with improved parameters
    images = []
    for _ in range(num_images):
        image = pipe(
            prompt,
            negative_prompt="low quality, blurry, distorted, unrealistic, amateur, bad anatomy",
            num_inference_steps=75,  # Increased for better quality
            guidance_scale=8.5,  # Slightly increased for better prompt adherence
            generator=generator,
            width=768,  # Higher resolution
            height=768
        ).images[0]
        images.append(image)
    
    return images

def save_generated_designs(images: List[Image.Image], output_dir: str = "generated_designs"):
    """Save the generated design images."""
    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(images):
        image.save(f"{output_dir}/generated_design_{i+1}.png")

def main():
    # Input text
    input_text = "blue waves pant"
    
    print("Loading knowledge graph...")
    graph_data = load_knowledge_graph()
    
    # Create NetworkX graph from the data
    G = nx.Graph()
    for node, data in graph_data['nodes']:
        G.add_node(node, **data)
    for u, v, data in graph_data['edges']:
        G.add_edge(u, v, **data)
    
    print("\nFinding relevant design paths...")
    paths = find_relevant_paths(G, input_text)
    
    if not paths:
        print("No relevant design paths found.")
        return
    
    # Present options and get user selection
    selected_path = present_path_options(G, paths)
    
    print("\nGenerating prompt from selected design path...")
    prompt = generate_prompt_from_path(input_text, selected_path)
    
    print("\nPrompt:", prompt)
    print("\nGenerating new designs...")
    generated_images = generate_designs(prompt)
    
    print("Saving generated designs...")
    save_generated_designs(generated_images)
    
    print("Process complete! Check the generated_designs directory for outputs.")

if __name__ == "__main__":
    main() 