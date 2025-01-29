import os
import h5py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import umap
from typing import Dict, List, Tuple

def load_embeddings(h5_file: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Load embeddings from H5 file."""
    embeddings = {}
    with h5py.File(h5_file, 'r') as f:
        paths = [p.decode('utf-8') for p in f['paths']]
        for path in paths:
            name = Path(path).stem
            embeddings[name] = f['embeddings'][name][:]
    return embeddings, paths

def create_umap_visualization(embeddings: Dict[str, np.ndarray], title: str, output_file: str):
    """Create UMAP visualization of embeddings."""
    # Prepare data for UMAP
    design_names = list(embeddings.keys())
    embedding_matrix = np.stack(list(embeddings.values()))
    
    # Create UMAP projection
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    embedding_2d = reducer.fit_transform(embedding_matrix)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.6)
    
    # Add labels for each point
    for i, name in enumerate(design_names):
        plt.annotate(name.split('-')[2:4], 
                    (embedding_2d[i, 0], embedding_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    plt.title(title)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"UMAP visualization saved to {output_file}")

def create_similarity_network(knowledge_base: Dict, output_file: str, min_similarity: float = 0.95):
    """Create network visualization of design similarities."""
    G = nx.Graph()
    
    # Add nodes
    for design_name in knowledge_base['designs'].keys():
        # Extract garment type from name
        garment_type = '-'.join(design_name.split('-')[2:4])
        G.add_node(design_name, type=garment_type)
    
    # Add edges for similar designs
    for relation in knowledge_base['relationships']:
        if relation['similarity'] >= min_similarity:
            G.add_edge(relation['design1'], relation['design2'], 
                      weight=relation['similarity'])
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Set up layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    node_colors = [hash(G.nodes[node]['type']) % 20 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, 
                          alpha=0.6, cmap=plt.cm.tab20)
    
    # Draw edges with width based on similarity
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4)
    
    # Add labels
    labels = {node: G.nodes[node]['type'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Fashion Design Similarity Network")
    plt.axis('off')
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Network visualization saved to {output_file}")

def load_fashion_clip_embeddings(csv_file: str) -> Dict[str, np.ndarray]:
    """Load Fashion-CLIP embeddings from CSV file."""
    df = pd.read_csv(csv_file)
    embeddings = {}
    
    for _, row in df.iterrows():
        design_name = row['design_name']
        embedding = row[[col for col in df.columns if col.startswith('dim_')]].values
        embeddings[design_name] = embedding
    
    return embeddings

def main():
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Load Fashion-CLIP embeddings from CSV
    print("Loading Fashion-CLIP embeddings...")
    fashion_clip_file = "embeddings/fashion_clip_embeddings.csv"
    fashion_clip_embeddings = load_fashion_clip_embeddings(fashion_clip_file)
    
    # Load description embeddings
    print("Loading description embeddings...")
    description_file = "embeddings/description_embeddings.h5"
    description_embeddings, _ = load_embeddings(description_file)
    
    # Load knowledge base
    print("Loading knowledge base...")
    with open("knowledge_base/fashion_knowledge_base.json", 'r') as f:
        knowledge_base = json.load(f)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_umap_visualization(
        fashion_clip_embeddings,
        "Fashion-CLIP Embeddings - UMAP Visualization",
        "visualizations/fashion_clip_umap.png"
    )
    
    create_umap_visualization(
        description_embeddings,
        "Description Embeddings - UMAP Visualization",
        "visualizations/description_umap.png"
    )
    
    create_similarity_network(
        knowledge_base,
        "visualizations/similarity_network.png"
    )
    
    print("\nVisualization complete! Check the 'visualizations' directory for output files.")

if __name__ == "__main__":
    main() 