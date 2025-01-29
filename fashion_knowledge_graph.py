import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import re

def extract_fashion_concepts(description: str) -> Dict[str, List[str]]:
    """Extract fashion-related concepts using regex patterns."""
    concepts = {
        'materials': [],
        'colors': [],
        'styles': [],
        'occasions': [],
        'design_elements': []
    }
    
    # Define fashion-related keywords
    material_patterns = r'\b(cotton|silk|wool|leather|denim|jersey|knit|fleece|canvas)\b'
    color_patterns = r'\b(black|white|red|blue|green|grey|taupe|sage|olive|beige|cream)\b'
    style_patterns = r'\b(formal|casual|elegant|minimalist|modern|classic|avant-garde)\b'
    occasion_patterns = r'\b(evening|formal|casual|office|party|editorial)\b'
    design_patterns = r'\b(sleeve|collar|pocket|button|zipper|hem|cuff|pleat|dart)\b'
    
    # Extract concepts using regex
    concepts['materials'] = list(set(re.findall(material_patterns, description.lower())))
    concepts['colors'] = list(set(re.findall(color_patterns, description.lower())))
    concepts['styles'] = list(set(re.findall(style_patterns, description.lower())))
    concepts['occasions'] = list(set(re.findall(occasion_patterns, description.lower())))
    concepts['design_elements'] = list(set(re.findall(design_patterns, description.lower())))
    
    return concepts

def calculate_similarity(desc1: Dict[str, List[str]], desc2: Dict[str, List[str]]) -> float:
    """Calculate similarity between two fashion descriptions."""
    total_similarity = 0
    weights = {
        'materials': 0.3,
        'colors': 0.2,
        'styles': 0.2,
        'occasions': 0.15,
        'design_elements': 0.15
    }
    
    for category, weight in weights.items():
        set1 = set(desc1[category])
        set2 = set(desc2[category])
        if set1 and set2:  # Only calculate if both sets have elements
            jaccard = len(set1 & set2) / len(set1 | set2)
            total_similarity += jaccard * weight
    
    return total_similarity

def create_knowledge_graph(descriptions: Dict[str, str], similarity_threshold: float = 0.3) -> nx.Graph:
    """Create a knowledge graph from fashion descriptions."""
    # Extract concepts from all descriptions
    print("Extracting fashion concepts...")
    concepts = {}
    for name, desc in tqdm(descriptions.items()):
        concepts[name] = extract_fashion_concepts(desc)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes (garments)
    for name in descriptions.keys():
        garment_type = '-'.join(name.split('-')[2:4])  # Extract garment type from name
        G.add_node(name, type=garment_type)
    
    # Add edges based on similarities
    print("Calculating similarities and creating edges...")
    for name1 in tqdm(descriptions.keys()):
        for name2 in descriptions.keys():
            if name1 < name2:  # Avoid duplicate edges
                similarity = calculate_similarity(concepts[name1], concepts[name2])
                if similarity > similarity_threshold:
                    G.add_edge(name1, name2, weight=similarity)
    
    return G

def visualize_knowledge_graph(G: nx.Graph, output_file: str):
    """Create a visual representation of the fashion knowledge graph."""
    plt.figure(figsize=(20, 20))
    
    # Calculate node sizes based on degree centrality
    centrality = nx.degree_centrality(G)
    node_sizes = [v * 5000 for v in centrality.values()]
    
    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create color map based on garment types
    garment_types = list(set(nx.get_node_attributes(G, 'type').values()))
    color_map = plt.cm.tab20(np.linspace(0, 1, len(garment_types)))
    type_to_color = dict(zip(garment_types, color_map))
    node_colors = [type_to_color[G.nodes[node]['type']] for node in G.nodes()]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.7)
    
    # Draw edges with varying thickness based on weight
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, 
                          width=edge_weights,
                          alpha=0.4,
                          edge_color='gray')
    
    # Add labels
    labels = {node: G.nodes[node]['type'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, 
                           font_size=8,
                           font_weight='bold')
    
    plt.title("Fashion Design Knowledge Graph", fontsize=16, pad=20)
    plt.axis('off')
    
    # Add legend for garment types
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, 
                                 label=garment_type,
                                 markersize=10)
                      for garment_type, color in type_to_color.items()]
    plt.legend(handles=legend_elements, 
              title="Garment Types",
              loc='center left',
              bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Knowledge graph visualization saved to {output_file}")

def main():
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Load descriptions
    descriptions = {}
    
    # Process Season 1
    season1_path = "Vizcom/Season_1"
    print("Processing Season 1 descriptions...")
    for image_file in sorted(os.listdir(season1_path)):
        if image_file.endswith(('.webp', '.jpg', '.jpeg', '.png')):
            name = Path(image_file).stem
            desc_file = f"descriptions/{name}.txt"
            try:
                if os.path.exists(desc_file):
                    with open(desc_file, 'r', encoding='utf-8') as f:
                        descriptions[name] = f.read()
                else:
                    print(f"Warning: Description file not found for {image_file}")
            except Exception as e:
                print(f"Error reading description for {image_file}: {str(e)}")
    
    # Process Season 2
    season2_path = "Vizcom/Season_2"
    print("Processing Season 2 descriptions...")
    for image_file in sorted(os.listdir(season2_path)):
        if image_file.endswith(('.webp', '.jpg', '.jpeg', '.png')):
            name = Path(image_file).stem
            desc_file = f"descriptions/{name}.txt"
            try:
                if os.path.exists(desc_file):
                    with open(desc_file, 'r', encoding='utf-8') as f:
                        descriptions[name] = f.read()
                else:
                    print(f"Warning: Description file not found for {image_file}")
            except Exception as e:
                print(f"Error reading description for {image_file}: {str(e)}")
    
    if not descriptions:
        print("Error: No descriptions found. Please run fashion_descriptions.py first.")
        return
    
    # Create and visualize knowledge graph
    print(f"\nCreating knowledge graph with {len(descriptions)} descriptions...")
    G = create_knowledge_graph(descriptions)
    
    print("\nVisualizing knowledge graph...")
    visualize_knowledge_graph(G, "visualizations/fashion_knowledge_graph_new.png")
    
    # Save graph data for future use
    print("\nSaving graph data...")
    graph_data = {
        'nodes': list(G.nodes(data=True)),
        'edges': list(G.edges(data=True))
    }
    with open("visualizations/fashion_graph_data.json", 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print("\nProcess complete! Check the visualizations directory for outputs.")

if __name__ == "__main__":
    main() 