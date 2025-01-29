import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import base64
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

class FashionPreprocessor:
    def __init__(self):
        """Initialize the fashion preprocessing pipeline."""
        load_dotenv()
        
        # Initialize paths
        self.descriptions_dir = "descriptions"
        self.visualizations_dir = "visualizations"
        os.makedirs(self.descriptions_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Initialize GPT-4V
        try:
            self.model = ChatOpenAI(model="gpt-4o", max_tokens=1000)
        except Exception as e:
            print(f"Error initializing model: {e}")
            print("Please ensure OPENAI_API_KEY is set correctly.")
            self.model = None
            
        # Fashion concept patterns
        self.patterns = {
            'materials': r'\b(cotton|silk|wool|leather|denim|jersey|knit|fleece|canvas)\b',
            'colors': r'\b(black|white|red|blue|green|grey|taupe|sage|olive|beige|cream)\b',
            'styles': r'\b(formal|casual|elegant|minimalist|modern|classic|avant-garde)\b',
            'occasions': r'\b(evening|formal|casual|office|party|editorial)\b',
            'design_elements': r'\b(sleeve|collar|pocket|button|zipper|hem|cuff|pleat|dart)\b'
        }
    
    def encode_image(self, image_path: str) -> Optional[str]:
        """Convert image to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def generate_description(self, image_path: str) -> Optional[str]:
        """Generate fashion description using GPT-4V."""
        if not self.model:
            return None
            
        base64_image = self.encode_image(image_path)
        if not base64_image:
            return None
            
        try:
            messages = [
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        },
                        {
                            "type": "text",
                            "text": """Please provide a comprehensive fashion design analysis including:

1. Overall Style & Category:
   - Garment type and silhouette
   - Design inspiration and aesthetic

2. Materials & Construction:
   - Fabric type and properties
   - Construction techniques
   - Notable stitching details

3. Design Elements:
   - Color palette and patterns
   - Unique design features
   - Hardware and closures
   - Pockets and functional elements

4. Fit & Silhouette:
   - Cut and drape
   - Proportions and length
   - Intended fit style

5. Styling & Versatility:
   - Suggested styling combinations
   - Occasion appropriateness
   - Seasonal relevance"""
                        }
                    ]
                )
            ]
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error generating description for {image_path}: {e}")
            return None

    def extract_concepts(self, description: str) -> Dict[str, List[str]]:
        """Extract fashion concepts using regex patterns."""
        concepts = {}
        for category, pattern in self.patterns.items():
            concepts[category] = list(set(re.findall(pattern, description.lower())))
        return concepts

    def calculate_similarity(self, desc1: Dict[str, List[str]], 
                           desc2: Dict[str, List[str]]) -> float:
        """Calculate weighted similarity between two fashion descriptions."""
        weights = {
            'materials': 0.3,
            'colors': 0.2,
            'styles': 0.2,
            'occasions': 0.15,
            'design_elements': 0.15
        }
        
        total_similarity = 0
        for category, weight in weights.items():
            set1 = set(desc1[category])
            set2 = set(desc2[category])
            if set1 and set2:
                jaccard = len(set1 & set2) / len(set1 | set2)
                total_similarity += jaccard * weight
        
        return total_similarity

    def create_knowledge_graph(self, descriptions: Dict[str, str], 
                             similarity_threshold: float = 0.3) -> nx.Graph:
        """Create fashion knowledge graph from descriptions."""
        # Extract concepts
        print("Extracting fashion concepts...")
        concepts = {
            name: self.extract_concepts(desc)
            for name, desc in tqdm(descriptions.items())
        }
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with garment type metadata
        for name in descriptions:
            garment_type = '-'.join(name.split('-')[2:4])
            G.add_node(name, type=garment_type)
        
        # Add edges based on similarities
        print("Calculating similarities and creating edges...")
        for name1 in tqdm(descriptions):
            for name2 in descriptions:
                if name1 < name2:
                    similarity = self.calculate_similarity(
                        concepts[name1], 
                        concepts[name2]
                    )
                    if similarity > similarity_threshold:
                        G.add_edge(name1, name2, weight=similarity)
        
        return G

    def visualize_graph(self, G: nx.Graph, output_file: str):
        """Visualize fashion knowledge graph."""
        plt.figure(figsize=(20, 20))
        
        # Calculate node sizes and positions
        centrality = nx.degree_centrality(G)
        node_sizes = [v * 5000 for v in centrality.values()]
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create color map for garment types
        garment_types = list(set(nx.get_node_attributes(G, 'type').values()))
        color_map = plt.cm.tab20(np.linspace(0, 1, len(garment_types)))
        type_to_color = dict(zip(garment_types, color_map))
        node_colors = [type_to_color[G.nodes[node]['type']] for node in G.nodes()]
        
        # Draw graph elements
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                             node_size=node_sizes, alpha=0.7)
        
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights,
                             alpha=0.4, edge_color='gray')
        
        labels = {node: G.nodes[node]['type'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color, label=garment_type,
                      markersize=10)
            for garment_type, color in type_to_color.items()
        ]
        plt.legend(handles=legend_elements, title="Garment Types",
                  loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.title("Fashion Design Knowledge Graph", fontsize=16, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def process_season(self, season_path: str) -> Dict[str, str]:
        """Process all designs in a season folder."""
        descriptions = {}
        
        if not os.path.exists(season_path):
            print(f"Warning: {season_path} does not exist")
            return descriptions
            
        print(f"\nProcessing {season_path}:")
        print("-" * 80)
        
        for image_file in sorted(os.listdir(season_path)):
            if image_file.endswith(('.webp', '.jpg', '.jpeg', '.png')):
                name = Path(image_file).stem
                desc_file = os.path.join(self.descriptions_dir, f"{name}.txt")
                
                if os.path.exists(desc_file):
                    # Load existing description
                    with open(desc_file, 'r', encoding='utf-8') as f:
                        descriptions[name] = f.read()
                    print(f"Loaded existing description for {image_file}")
                else:
                    # Generate new description
                    print(f"Generating description for {image_file}")
                    image_path = os.path.join(season_path, image_file)
                    description = self.generate_description(image_path)
                    
                    if description:
                        descriptions[name] = description
                        # Save description
                        with open(desc_file, 'w', encoding='utf-8') as f:
                            f.write(description)
                        print(f"Saved description to {desc_file}")
        
        return descriptions

    def process_all(self):
        """Process all seasons and create knowledge graph."""
        # Process both seasons
        descriptions = {}
        for season in ["Season_1", "Season_2"]:
            season_path = os.path.join("Vizcom", season)
            season_descriptions = self.process_season(season_path)
            descriptions.update(season_descriptions)
        
        if not descriptions:
            print("Error: No descriptions found or generated.")
            return
        
        # Create and visualize knowledge graph
        print(f"\nCreating knowledge graph with {len(descriptions)} descriptions...")
        G = self.create_knowledge_graph(descriptions)
        
        print("\nVisualizing knowledge graph...")
        self.visualize_graph(G, 
            os.path.join(self.visualizations_dir, "fashion_knowledge_graph.png"))
        
        # Save graph data
        print("\nSaving graph data...")
        graph_data = {
            'nodes': list(G.nodes(data=True)),
            'edges': list(G.edges(data=True))
        }
        with open(os.path.join(self.visualizations_dir, "fashion_graph_data.json"), 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print("\nPreprocessing complete! Check descriptions and visualizations directories.")

def main():
    preprocessor = FashionPreprocessor()
    preprocessor.process_all()

if __name__ == "__main__":
    main() 