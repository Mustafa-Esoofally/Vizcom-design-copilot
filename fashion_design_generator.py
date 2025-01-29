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
from datetime import datetime

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
    category_to_nodes = defaultdict(set)
    node_elements = {}
    
    # Define common fashion elements
    material_keywords = {'cotton', 'silk', 'wool', 'leather', 'denim', 'jersey', 'knit', 'fleece', 'canvas'}
    color_keywords = {'black', 'white', 'red', 'blue', 'green', 'grey', 'taupe', 'sage', 'olive', 'beige', 'cream'}
    style_keywords = {'formal', 'casual', 'elegant', 'minimalist', 'avant-garde', 'modern', 'classic'}
    
    # Index all nodes by their elements and categories
    for node in G.nodes():
        elements = get_node_fashion_elements(node)
        node_elements[node] = elements
        node_type = G.nodes[node]['type'].lower()
        
        # Add node type as a category
        category_to_nodes[node_type].add(node)
        
        # Index nodes by their elements
        for category, items in elements.items():
            for item in items:
                item_lower = item.lower()
                if item_lower in material_keywords:
                    element_to_nodes[f"material_{item_lower}"].add(node)
                elif item_lower in color_keywords:
                    element_to_nodes[f"color_{item_lower}"].add(node)
                elif item_lower in style_keywords:
                    element_to_nodes[f"style_{item_lower}"].add(node)
                
                # Add to category index
                category_to_nodes[category].add(node)
    
    # Find relevant starting points based on input keywords
    start_nodes = set()
    for node in G.nodes():
        node_type = G.nodes[node]['type'].lower()
        elements = node_elements[node]
        
        # Match by type or elements
        if any(keyword in node_type for keyword in keywords) or \
           any(keyword in ' '.join(str(item) for items in elements.values() for item in items).lower() for keyword in keywords):
            start_nodes.add(node)
    
    # Create diverse paths based on category relationships
    paths = []
    seen_combinations = set()
    
    for start_node in start_nodes:
        start_elements = node_elements[start_node]
        
        # Get related categories based on start node
        related_materials = set(item.lower() for item in start_elements.get('materials', []))
        related_colors = set(item.lower() for item in start_elements.get('colors', []))
        related_styles = set(item.lower() for item in start_elements.get('styles', []))
        
        # Find nodes with complementary elements
        for material in (material_keywords & related_materials):
            material_nodes = element_to_nodes[f"material_{material}"] - {start_node}
            if not material_nodes:
                continue
            
            for color in (color_keywords & related_colors):
                color_nodes = element_to_nodes[f"color_{color}"] - {start_node} - material_nodes
                if not color_nodes:
                    continue
                
                for style in (style_keywords & related_styles):
                    style_nodes = element_to_nodes[f"style_{style}"] - {start_node} - material_nodes - color_nodes
                    if not style_nodes:
                        continue
                    
                    # Create path with related elements
                    combination = (material, color, style)
                    if combination in seen_combinations:
                        continue
                    
                    # Find intermediate nodes that connect the elements
                    intermediate_nodes = set()
                    for node in G.nodes():
                        if node not in {start_node} | material_nodes | color_nodes | style_nodes:
                            node_elems = node_elements[node]
                            if any(m in node_elems.get('materials', []) for m in related_materials) or \
                               any(c in node_elems.get('colors', []) for c in related_colors) or \
                               any(s in node_elems.get('styles', []) for s in related_styles):
                                intermediate_nodes.add(node)
                    
                    if intermediate_nodes:
                        # Create path with intermediate nodes
                        path = [
                            start_node,
                            min(intermediate_nodes),  # Connecting node
                            min(material_nodes | color_nodes),  # Material/color focused node
                            min(style_nodes)  # Style focused node
                        ]
                        
                        if len(set(path)) == len(path):  # Ensure all nodes are unique
                            paths.append(path)
                            seen_combinations.add(combination)
                            
                            if len(paths) >= max_paths:
                                return paths[:max_paths]
    
    # If we don't have enough paths, try creating alternative paths
    if len(paths) < max_paths:
        for start_node in start_nodes:
            # Get nodes with high similarity to start_node
            similar_nodes = []
            for node in G.nodes():
                if node != start_node:
                    similarity = len(set(str(node_elements[node])) & set(str(node_elements[start_node]))) / \
                               len(set(str(node_elements[node])) | set(str(node_elements[start_node])))
                    similar_nodes.append((node, similarity))
            
            # Sort by similarity
            similar_nodes.sort(key=lambda x: x[1], reverse=True)
            
            # Create alternative paths using similar nodes
            for similar_node, _ in similar_nodes[:3]:
                if len(paths) >= max_paths:
                    break
                    
                # Find connecting nodes
                connecting_nodes = set(G.neighbors(similar_node)) - {start_node, similar_node}
                if connecting_nodes:
                    path = [
                        start_node,
                        similar_node,
                        min(connecting_nodes),
                        max(connecting_nodes)
                    ]
                    
                    if len(set(path)) == len(path) and path not in paths:
                        paths.append(path)
    
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

def generate_prompt_from_path(input_text: str, path_info: Dict) -> Dict[str, str]:
    """Generate detailed prompts using the selected path's design elements."""
    # Parse input elements
    input_words = input_text.lower().split()
    
    # Base concept
    base_concept = {
        'garment_type': next((word for word in input_words if 'pant' in word), 'pants'),
        'primary_color': next((word for word in input_words if word in ['blue', 'black', 'white', 'red', 'green']), None),
        'style_keyword': next((word for word in input_words if word not in ['pant', 'pants', 'blue', 'black', 'white']), None)
    }
    
    # Create main prompt
    main_prompt = f"Professional fashion photography of high-end designer {base_concept['garment_type']}"
    if base_concept['primary_color']:
        main_prompt += f" in {base_concept['primary_color']}"
    if base_concept['style_keyword']:
        main_prompt += f" with {base_concept['style_keyword']} pattern"
    
    # Add key elements by category
    for category, items in path_info['elements'].items():
        if items and category != 'type':
            # Select most relevant items (up to 2 per category)
            key_items = items[:2]
            if category == 'materials':
                main_prompt += f", made from {' and '.join(key_items)}"
            elif category == 'design_elements':
                main_prompt += f", featuring {' and '.join(key_items)}"
    
    # Add style elements
    if path_info['styles']:
        key_styles = path_info['styles'][:2]
        main_prompt += f", {' and '.join(key_styles)} style"
    
    # Create style prompt
    style_prompt = "Highly detailed fashion design, professional studio lighting, "
    style_prompt += "high-end fashion magazine quality, crisp details, "
    style_prompt += "professional fashion photography, full body shot, "
    style_prompt += "clean background, fashion lookbook style"
    
    # Create negative prompt
    negative_prompt = "low quality, blurry, distorted, unrealistic, amateur, "
    negative_prompt += "bad anatomy, deformed, disfigured, poorly drawn face, "
    negative_prompt += "mutated, extra limbs, ugly, poorly drawn hands, "
    negative_prompt += "missing fingers, extra fingers, floating limbs, "
    negative_prompt += "disconnected limbs, mutation, deformed hands, "
    negative_prompt += "out of frame, truncated, worst quality"
    
    return {
        'prompt': main_prompt,
        'style_prompt': style_prompt,
        'negative_prompt': negative_prompt
    }

def generate_designs(prompts: Dict[str, str], num_images: int = 5) -> List[Image.Image]:
    """Generate new fashion designs using Stable Diffusion with enhanced control."""
    # Initialize Stable Diffusion pipeline
    model_id = "runwayml/stable-diffusion-v1-5"  # Using a more stable model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    # Enable attention slicing for memory efficiency
    pipe.enable_attention_slicing()
    
    # Combine prompts
    full_prompt = f"{prompts['prompt']}. {prompts['style_prompt']}"
    
    # Generation parameters
    params = {
        "prompt": full_prompt,
        "negative_prompt": prompts['negative_prompt'],
        "num_inference_steps": 100,  # Increased for better quality
        "guidance_scale": 9.0,  # Increased for better prompt adherence
        "width": 768,  # Balanced resolution
        "height": 768,
        "num_images_per_prompt": 1
    }
    
    # Generate images with different seeds
    images = []
    for i in range(num_images):
        try:
            # Set different seed for each generation
            generator = torch.Generator(device="cuda").manual_seed(42 + i)
            result = pipe(**params, generator=generator)
            images.append(result.images[0])
        except Exception as e:
            print(f"Error generating image {i+1}: {str(e)}")
            continue
    
    return images

def save_generated_designs(images: List[Image.Image], prompts: Dict[str, str], 
                         output_dir: str = "generated_designs"):
    """Save the generated design images with their prompts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prompts
    prompt_file = os.path.join(output_dir, "generation_prompts.json")
    with open(prompt_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    # Save images with metadata
    for i, image in enumerate(images):
        # Save high-res image
        image_path = os.path.join(output_dir, f"generated_design_{i+1}.png")
        image.save(image_path, quality=95)
        
        # Save metadata
        meta_path = os.path.join(output_dir, f"generated_design_{i+1}_meta.json")
        metadata = {
            "image_number": i + 1,
            "resolution": f"{image.size[0]}x{image.size[1]}",
            "prompts": prompts,
            "timestamp": datetime.now().isoformat()
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

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
    
    print("\nGenerating prompts from selected design path...")
    prompts = generate_prompt_from_path(input_text, selected_path)
    
    print("\nMain Prompt:", prompts['prompt'])
    print("Style Prompt:", prompts['style_prompt'])
    print("\nGenerating new designs...")
    generated_images = generate_designs(prompts)
    
    print("Saving generated designs...")
    save_generated_designs(generated_images, prompts)
    
    print("Process complete! Check the generated_designs directory for outputs.")

if __name__ == "__main__":
    main() 