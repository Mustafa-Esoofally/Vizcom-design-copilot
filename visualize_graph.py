import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph
import json
import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from pathlib import Path
import h5py
from tqdm import tqdm

class FashionCLIPEmbedder:
    def __init__(self):
        """Initialize Fashion-CLIP model and processor."""
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], output_file: str):
        """Save embeddings to HDF5 format."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, 'w') as f:
            # Create a group for embeddings
            emb_group = f.create_group('embeddings')
            
            # Save each embedding
            for path, embedding in embeddings.items():
                # Use base filename as key
                key = Path(path).stem
                emb_group.create_dataset(key, data=embedding)
            
            # Save metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['num_embeddings'] = len(embeddings)
            meta_group.attrs['embedding_dim'] = next(iter(embeddings.values())).shape[0]
            
            # Save path mappings
            path_group = f.create_group('paths')
            for i, path in enumerate(embeddings.keys()):
                path_group.attrs[f'path_{i}'] = str(path)
    
    def load_embeddings(self, input_file: str) -> Dict[str, np.ndarray]:
        """Load embeddings from HDF5 format."""
        embeddings = {}
        with h5py.File(input_file, 'r') as f:
            emb_group = f['embeddings']
            path_group = f['paths']
            
            # Reconstruct path to embedding mapping
            path_mapping = {i: path_group.attrs[f'path_{i}'] for i in range(len(path_group.attrs))}
            
            # Load embeddings
            for key in emb_group.keys():
                # Find original path
                original_path = next(path for path in path_mapping.values() if Path(path).stem == key)
                embeddings[original_path] = emb_group[key][:]
        
        return embeddings

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Get Fashion-CLIP embedding for an image."""
        try:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy()[0]
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

def compute_fashion_similarity(embedder: FashionCLIPEmbedder, image_paths: List[str], cache_file: str = None) -> Dict[tuple, float]:
    """Compute pairwise similarities between fashion images using Fashion-CLIP embeddings."""
    # Try to load cached embeddings
    embeddings = {}
    if cache_file and os.path.exists(cache_file):
        print("Loading cached embeddings...")
        embeddings = embedder.load_embeddings(cache_file)
        print(f"Loaded {len(embeddings)} cached embeddings")
    
    # Compute missing embeddings
    missing_paths = [p for p in image_paths if p not in embeddings]
    if missing_paths:
        print(f"Computing embeddings for {len(missing_paths)} new images...")
        for path in tqdm(missing_paths):
            embedding = embedder.get_image_embedding(path)
            if embedding is not None:
                embeddings[path] = embedding
        
        # Save updated embeddings
        if cache_file:
            print("Saving embeddings to cache...")
            embedder.save_embeddings(embeddings, cache_file)
    
    # Compute pairwise similarities
    similarities = {}
    print("Computing pairwise similarities...")
    for i, (path1, emb1) in enumerate(tqdm(embeddings.items())):
        for path2, emb2 in list(embeddings.items())[i+1:]:
            similarity = float(np.dot(emb1, emb2))
            similarities[(path1, path2)] = similarity
    
    return similarities

def create_fashion_knowledge_graph(similarities: Dict[tuple, float], threshold: float = 0.7) -> nx.Graph:
    """Create a knowledge graph from fashion similarities."""
    graph = nx.Graph()
    
    # Add edges for similar designs
    for (path1, path2), similarity in similarities.items():
        if similarity >= threshold:
            # Extract design names from paths
            name1 = Path(path1).stem
            name2 = Path(path2).stem
            
            # Add nodes with metadata
            if not graph.has_node(name1):
                graph.add_node(name1, metadata={'image_path': path1})
            if not graph.has_node(name2):
                graph.add_node(name2, metadata={'image_path': path2})
            
            # Add edge with similarity weight
            graph.add_edge(name1, name2, weight=similarity)
    
    return graph

def create_custom_colormap():
    """Create a custom colormap for edge weights."""
    colors = ['#E6F3FF', '#0066CC']  # Light blue to dark blue
    return LinearSegmentedColormap.from_list('custom_blue', colors)

def load_graph_data(graph_name: str = "fashion_knowledge_graph") -> Dict[str, Any]:
    """Load the saved knowledge graph data."""
    try:
        # Load graph statistics
        with open(f"{graph_name}_stats.json", "r") as f:
            stats = json.load(f)
        
        # Load NetworkX graph
        graph = nx.read_gexf(f"{graph_name}.gexf")
        
        # Initialize empty RDF graph (since we don't have the serialized version)
        rdf_graph = Graph()
        
        return {
            'stats': stats,
            'graph': graph,
            'rdf_graph': rdf_graph
        }
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure the knowledge graph files exist in the current directory.")
        return None

def visualize_network_graph(graph: nx.Graph):
    """Visualize the NetworkX graph structure with enhanced formatting."""
    # Set figure size and style
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Create custom colormap
    edge_cmap = create_custom_colormap()
    
    # Use force-directed layout with adjusted parameters
    pos = nx.spring_layout(
        graph,
        k=1/np.sqrt(len(graph.nodes())),  # Optimal distance between nodes
        iterations=50,  # More iterations for better layout
        seed=42  # For reproducibility
    )
    
    # Draw edges with weights as colors and varying widths
    edges = graph.edges()
    if edges:
        weights = [graph[u][v].get('weight', 0.5) for u, v in edges]
        # Normalize weights for width calculation
        widths = [2 + 3 * w for w in weights]
        
        nx.draw_networkx_edges(
            graph, pos,
            edge_color=weights,
            edge_cmap=edge_cmap,
            width=widths,
            alpha=0.7,
            edge_vmin=0,
            edge_vmax=1,
            ax=ax
        )
    
    # Calculate node sizes based on degree centrality
    centrality = nx.degree_centrality(graph)
    node_sizes = [3000 * (0.5 + centrality[node]) for node in graph.nodes()]
    
    # Draw nodes with size variation and custom style
    nx.draw_networkx_nodes(
        graph, pos,
        node_size=node_sizes,
        node_color='#FFE5CC',  # Light orange
        edgecolors='#FF9933',  # Darker orange
        linewidths=2,
        alpha=0.9,
        ax=ax
    )
    
    # Add labels with custom formatting
    labels = {node: '\n'.join(node.split('-')[:2]) for node in graph.nodes()}
    nx.draw_networkx_labels(
        graph, pos,
        labels=labels,
        font_size=8,
        font_weight='bold',
        font_family='sans-serif',
        ax=ax
    )
    
    # Add title
    ax.set_title(
        "Fashion Design Knowledge Graph (Fashion-CLIP Embeddings)\nNode size indicates connectivity, edge thickness indicates style similarity",
        fontsize=16,
        pad=20,
        fontweight='bold'
    )
    
    # Add colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Style Similarity Score', orientation='horizontal', pad=0.05, aspect=50)
    
    ax.axis('off')
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(
        'knowledge_graph_visualization.png',
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close()

def visualize_clusters(graph: nx.Graph):
    """Create additional visualization showing design clusters."""
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Detect communities using Louvain method
    try:
        import community
        communities = community.best_partition(graph)
        num_communities = len(set(communities.values()))
        
        # Position nodes using force-directed layout
        pos = nx.spring_layout(graph, k=1/np.sqrt(len(graph.nodes())), iterations=50)
        
        # Draw nodes colored by community
        colors = plt.cm.rainbow(np.linspace(0, 1, num_communities))
        
        for node in graph.nodes():
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=[node],
                node_size=2000,
                node_color=[colors[communities[node]]],
                alpha=0.8,
                edgecolors='white',
                linewidths=1.5,
                ax=ax
            )
        
        # Draw edges with transparency
        nx.draw_networkx_edges(
            graph, pos,
            alpha=0.2,
            edge_color='gray',
            ax=ax
        )
        
        # Add labels
        labels = {node: '\n'.join(node.split('-')[:2]) for node in graph.nodes()}
        nx.draw_networkx_labels(
            graph, pos,
            labels=labels,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        
        ax.set_title(
            "Fashion Design Clusters\nColors indicate design communities",
            fontsize=16,
            pad=20,
            fontweight='bold'
        )
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save cluster visualization
        plt.savefig(
            'design_clusters_visualization.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close()
        
        return communities
    except ImportError:
        print("Note: python-louvain package not installed. Skipping cluster visualization.")
        return None

def print_graph_summary(data: Dict[str, Any]):
    """Print a detailed summary of the knowledge graph."""
    if not data:
        return
    
    stats = data['stats']
    graph = data['graph']
    
    print("\n=== Knowledge Graph Summary ===")
    print(f"\nBasic Statistics:")
    print(f"Number of nodes: {stats['nodes']}")
    print(f"Number of edges: {stats['edges']}")
    print(f"Number of RDF triples: {stats['triples']}")
    
    print("\nNode Categories:")
    node_types = {}
    for node, attrs in graph.nodes(data=True):
        if 'metadata' in attrs and isinstance(attrs['metadata'], dict):
            node_type = attrs['metadata'].get('product_type', 'unknown')
        else:
            node_type = 'unknown'
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    for node_type, count in node_types.items():
        print(f"- {node_type}: {count} nodes")
    
    print("\nTop Connected Designs:")
    degree_centrality = nx.degree_centrality(graph)
    top_designs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    for design, centrality in top_designs:
        print(f"- {design}: {centrality:.3f} centrality score")
    
    print("\nGraph Density:", nx.density(graph))
    print("Average Clustering Coefficient:", nx.average_clustering(graph))
    
    # Print strongest connections
    print("\nStrongest Design Relationships:")
    edges = [(u, v, d.get('weight', 0)) for u, v, d in graph.edges(data=True)]
    strongest_edges = sorted(edges, key=lambda x: x[2], reverse=True)[:5]
    for u, v, w in strongest_edges:
        print(f"- {u} ←→ {v} (similarity: {w:.3f})")

def export_graph_data(data: Dict[str, Any], output_dir: str = "graph_analysis"):
    """Export graph data to various formats for analysis."""
    if not data:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Export node data to CSV
    node_data = []
    for node, attrs in data['graph'].nodes(data=True):
        metadata = attrs.get('metadata', {})
        if isinstance(metadata, dict):
            node_info = {
                'node_id': node,
                'type': metadata.get('product_type', 'unknown'),
                'description': attrs.get('description', ''),
                'degree': data['graph'].degree(node)
            }
            node_data.append(node_info)
    
    pd.DataFrame(node_data).to_csv(
        os.path.join(output_dir, 'node_analysis.csv'),
        index=False
    )
    
    # Export edge data
    edge_data = []
    for u, v, attrs in data['graph'].edges(data=True):
        edge_info = {
            'source': u,
            'target': v,
            'weight': attrs.get('weight', 0),
            'type': attrs.get('type', 'similarity')
        }
        edge_data.append(edge_info)
    
    pd.DataFrame(edge_data).to_csv(
        os.path.join(output_dir, 'edge_analysis.csv'),
        index=False
    )
    
    # Export graph metrics
    metrics = {
        'density': nx.density(data['graph']),
        'avg_clustering': nx.average_clustering(data['graph']),
        'avg_degree': sum(dict(data['graph'].degree()).values()) / len(data['graph']),
        'diameter': nx.diameter(data['graph']) if nx.is_connected(data['graph']) else 'disconnected',
        'connected_components': nx.number_connected_components(data['graph'])
    }
    
    with open(os.path.join(output_dir, 'graph_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    # Initialize Fashion-CLIP embedder
    embedder = FashionCLIPEmbedder()
    
    # Get all image paths from the Season_1 directory
    image_dir = "Vizcom/Season_1"
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.webp']:
        image_paths.extend([str(p) for p in Path(image_dir).glob(ext)])
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images. Computing Fashion-CLIP embeddings...")
    
    # Set up embedding cache file
    os.makedirs("embeddings", exist_ok=True)
    cache_file = "embeddings/fashion_clip_embeddings.h5"
    
    # Compute similarities using Fashion-CLIP with caching
    similarities = compute_fashion_similarity(embedder, image_paths, cache_file=cache_file)
    
    # Create knowledge graph
    graph = create_fashion_knowledge_graph(similarities)
    
    print("\nGraph created successfully!")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_network_graph(graph)
    communities = visualize_clusters(graph)
    
    # Print detailed summary
    print_graph_summary({'graph': graph, 'stats': {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'triples': len(similarities)
    }})
    
    # Export data for further analysis
    export_graph_data({'graph': graph, 'stats': {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'triples': len(similarities)
    }})
    
    print("\nAnalysis complete!")
    print("- Main graph visualization saved as 'knowledge_graph_visualization.png'")
    print("- Fashion-CLIP embeddings saved to 'embeddings/fashion_clip_embeddings.h5'")
    if communities is not None:
        print("- Cluster visualization saved as 'design_clusters_visualization.png'")
    print("- Detailed data exported to 'graph_analysis' directory")

if __name__ == "__main__":
    main() 