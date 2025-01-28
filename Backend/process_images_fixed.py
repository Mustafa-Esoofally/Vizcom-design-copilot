import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, OwlViTForObjectDetection, OwlViTProcessor
import logging
import networkx as nx
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """Structure for memory storage"""
    timestamp: datetime
    content_type: str
    data: Dict
    source: str
    relevance_score: float
    status: str = "active"
    tags: List[str] = None
    priority: int = 1
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": str(self.timestamp),
            "content_type": self.content_type,
            "data": self.data,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "status": self.status,
            "tags": self.tags or [],
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryItem':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class DesignMemory:
    """Memory management for design processing"""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_file = self.cache_dir / "design_memory.json"
        self.items = []
        self._load_memory()
    
    def _load_memory(self):
        """Load memory from cache"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.items = [MemoryItem.from_dict(item) for item in data]
                    logger.info(f"Loaded {len(self.items)} items from memory")
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            self.items = []
    
    def add_item(self, content_type: str, data: Dict, source: str, 
                 relevance_score: float = 1.0, tags: List[str] = None):
        """Add new memory item"""
        item = MemoryItem(
            timestamp=datetime.now(),
            content_type=content_type,
            data=data,
            source=source,
            relevance_score=relevance_score,
            tags=tags
        )
        self.items.append(item)
        self._save_memory()
    
    def _save_memory(self):
        """Save memory to cache"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump([item.to_dict() for item in self.items], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

class DesignKnowledgeGraph:
    """Knowledge graph for design elements"""
    def __init__(self):
        self.graph = nx.Graph()
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize basic design knowledge"""
        # Add nodes for different categories
        categories = {
            "style": ["modern", "minimal", "technical", "sporty"],
            "material": ["fabric", "metal", "plastic", "elastic"],
            "component": ["collar", "zipper", "pocket", "seam", "logo"]
        }
        
        for category, elements in categories.items():
            for element in elements:
                self.graph.add_node(element, category=category)
        
        # Add basic relationships
        relationships = [
            ("collar", "fabric", 0.8),
            ("zipper", "metal", 0.9),
            ("pocket", "fabric", 0.8),
            ("seam", "fabric", 0.9),
            ("logo", "fabric", 0.7),
            ("modern", "minimal", 0.7),
            ("technical", "sporty", 0.8)
        ]
        
        for elem1, elem2, weight in relationships:
            self.graph.add_edge(elem1, elem2, weight=weight)
    
    def get_related_elements(self, element: str, max_distance: int = 2) -> List[Tuple[str, float]]:
        """Get related elements within a certain distance"""
        if element not in self.graph:
            return []
        
        related = []
        for node in self.graph.nodes():
            if node != element:
                try:
                    distance = nx.shortest_path_length(
                        self.graph, element, node, weight='weight'
                    )
                    if distance <= max_distance:
                        strength = 1.0 / (1.0 + distance)
                        related.append((node, strength))
                except nx.NetworkXNoPath:
                    continue
        
        return sorted(related, key=lambda x: x[1], reverse=True)

class DesignProcessor:
    def __init__(self):
        """Initialize models and processors"""
        logger.info("Loading models...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        
        # Initialize image generation model
        self.image_generator = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize memory system
        self.memory = DesignMemory(Path("cache"))
        
        # Initialize knowledge graph
        self.knowledge_graph = DesignKnowledgeGraph()
        
        # Initialize design elements and properties
        self._initialize_design_attributes()
    
    def _initialize_design_attributes(self):
        """Initialize design elements and properties dynamically"""
        # Core design elements that can be detected
        self.design_elements = {
            "hood": ["adjustable", "lined", "structured"],
            "pocket": ["spacious", "reinforced", "functional"],
            "cuffs": ["elastic", "durable", "fitted"],
            "hem": ["elastic", "structured", "fitted"],
            "zipper": ["premium", "durable", "smooth"],
            "logo": ["embroidered", "reflective", "placed"],
            "panels": ["reinforced", "articulated", "styled"],
            "seams": ["ventilated", "stretch", "fitted"]
        }
        
        # Material properties
        self.material_properties = {
            "moisture-wicking": ["comfort", "performance", "dry"],
            "stretch": ["mobility", "fit", "comfort"],
            "interior": ["warmth", "soft", "cozy"],
            "lightweight": ["breathable", "layerable", "comfortable"],
            "durable": ["durable", "premium", "quality"]
        }
    
    def analyze_style(self, image: Image) -> Dict:
        """Analyze style using CLIP model with dynamic categories"""
        # Define style queries based on design aspects
        style_queries = [
            "This is a technical and performance focused design",
            "This is a minimal and clean design",
            "This is an urban and modern design",
            "This is a premium and sophisticated design",
            "This is an innovative and functional design",
            "This is a classic and traditional design",
            "This is a sporty and dynamic design",
            "This is a casual and comfortable design"
        ]
        
        # Process image with CLIP
        inputs = self.clip_processor(
            images=image,
            text=style_queries,
            return_tensors="pt",
            padding=True
        )
        
        outputs = self.clip_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits_per_image[0], dim=0)
        
        # Get style predictions
        style_scores = []
        for query, score in zip(style_queries, probs):
            if score > 0.2:  # Lower threshold for sensitivity
                style_type = query.split("This is a")[-1].split("design")[0].strip()
                style_scores.append({
                    "style": style_type,
                    "confidence": float(score),
                    "attributes": self._extract_style_attributes(style_type)
                })
        
        return sorted(style_scores, key=lambda x: x["confidence"], reverse=True)
    
    def _extract_style_attributes(self, style_type: str) -> List[str]:
        """Extract relevant attributes based on style type"""
        # Split compound style types and extract key attributes
        attributes = []
        style_words = style_type.split(" and ")
        for word in style_words:
            word = word.strip()
            if "technical" in word or "performance" in word:
                attributes.extend(["functional", "innovative", "ergonomic"])
            elif "minimal" in word or "clean" in word:
                attributes.extend(["refined", "structured", "balanced"])
            elif "urban" in word or "modern" in word:
                attributes.extend(["contemporary", "dynamic", "bold"])
            elif "premium" in word or "sophisticated" in word:
                attributes.extend(["quality", "luxurious", "refined"])
            elif "innovative" in word or "functional" in word:
                attributes.extend(["advanced", "practical", "efficient"])
            elif "classic" in word or "traditional" in word:
                attributes.extend(["timeless", "reliable", "authentic"])
            elif "sporty" in word or "dynamic" in word:
                attributes.extend(["active", "energetic", "fluid"])
            elif "casual" in word or "comfortable" in word:
                attributes.extend(["relaxed", "versatile", "easy-going"])
        
        return list(set(attributes))  # Remove duplicates

    def process_design(self, image_path: Path) -> Dict:
        """Process a single design image following the three-phase workflow"""
        try:
            logger.info(f"\nProcessing {image_path.name}...")
            
            # INPUT PHASE
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # PROCESS PHASE
            # Style Analysis using image model
            style_scores = self.analyze_style(image)
            
            # Element Detection with knowledge graph integration
            elements = {}
            for element, attributes in self.design_elements.items():
                element_inputs = self.owlvit_processor(
                    images=image,
                    text=[element],
                    return_tensors="pt"
                )
                
                outputs = self.owlvit_model(**element_inputs)
                target_sizes = torch.Tensor([image.size[::-1]])
                results = self.owlvit_processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=0.03  # Sensitive detection
                )[0]
                
                if len(results["scores"]) > 0:
                    max_score_idx = results["scores"].argmax()
                    score = float(results["scores"][max_score_idx])
                    if score > 0.25:
                        # Get related elements from knowledge graph
                        related = self.knowledge_graph.get_related_elements(element)
                        elements[element] = {
                            "box": results["boxes"][max_score_idx].tolist(),
                            "score": score,
                            "attributes": attributes,
                            "related_elements": [
                                {"name": rel_name, "strength": rel_strength}
                                for rel_name, rel_strength in related
                            ]
                        }
            
            # OUTPUT PHASE
            # Compile results with relationships
            results = {
                "filename": image_path.name,
                "style_analysis": {
                    "dominant_styles": style_scores,
                    "design_score": float(sum(s["confidence"] for s in style_scores) / len(style_scores)),
                    "style_relationships": [
                        {
                            "primary": style["style"],
                            "attributes": style["attributes"],
                            "related_styles": [
                                s["style"] for s in style_scores 
                                if s["confidence"] > 0.3 and s["style"] != style["style"]
                            ]
                        }
                        for style in style_scores if style["confidence"] > 0.4
                    ]
                },
                "element_analysis": {
                    "detected_elements": elements,
                    "element_relationships": [
                        {
                            "element": elem_name,
                            "related_elements": elem_data["related_elements"][:5]
                        }
                        for elem_name, elem_data in elements.items()
                    ]
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Store results in memory
            self.memory.add_item(
                content_type="design_analysis",
                data=results,
                source=str(image_path),
                tags=["analysis", "style", "elements"]
            )
            
            # Print detailed analysis
            self._print_analysis_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            return {"error": str(e)}

    def _print_analysis_results(self, results: Dict):
        """Print formatted analysis results"""
        print(f"\n{'='*50}")
        print(f"Detailed Analysis for {results['filename']}")
        print(f"{'='*50}")
        
        print("\nStyle Analysis:")
        print("--------------")
        for style in results["style_analysis"]["dominant_styles"]:
            print(f"  - {style['style']} (confidence: {style['confidence']:.2f})")
            if style["attributes"]:
                print(f"    Attributes: {', '.join(style['attributes'])}")
        
        if "style_relationships" in results["style_analysis"]:
            print("\nStyle Relationships:")
            for rel in results["style_analysis"]["style_relationships"]:
                print(f"  • {rel['primary']}")
                if rel["related_styles"]:
                    print(f"    Related styles: {', '.join(rel['related_styles'])}")
        
        print("\nElement Analysis:")
        print("-----------------")
        if results["element_analysis"]["detected_elements"]:
            for elem_name, elem_data in results["element_analysis"]["detected_elements"].items():
                print(f"\n  • {elem_name} (confidence: {elem_data['score']:.2f})")
                print(f"    Attributes: {', '.join(elem_data['attributes'])}")
                if elem_data["related_elements"]:
                    print("    Related elements:")
                    for rel in elem_data["related_elements"][:5]:
                        print(f"      - {rel['name']} (strength: {rel['strength']:.2f})")
        else:
            print("  No elements detected with current confidence threshold")
        
        print(f"\nTimestamp: {results['timestamp']}")
        print(f"{'='*50}\n")

    def generate_new_design(self, variation_name: str) -> Dict:
        """Generate a new hoodie design based on specified variation"""
        variation = self.design_variations[variation_name]
        
        # Compile design specification
        design = {
            "name": f"Light Blue {variation_name.title()} Hoodie",
            "base_color": "light blue",
            "style_category": variation["style"],
            "design_elements": [],
            "material_features": [],
            "construction_details": []
        }
        
        # Add detailed elements
        for element in variation["elements"]:
            if element in self.design_elements:
                design["design_elements"].append({
                    "name": element,
                    "attributes": self.design_elements[element],
                    "placement": self._get_element_placement(element)
                })
        
        # Add material features
        for feature in variation["features"]:
            if feature in self.material_properties:
                design["material_features"].append({
                    "name": feature,
                    "properties": self.material_properties[feature]
                })
        
        # Add construction details based on style
        design["construction_details"] = self._get_construction_details(variation["style"])
        
        return design
    
    def _get_element_placement(self, element: str) -> Dict:
        """Define placement details for design elements"""
        placements = {
            "hood": {"position": "top", "alignment": "centered"},
            "kangaroo pocket": {"position": "front-lower", "alignment": "centered"},
            "ribbed cuffs": {"position": "sleeves-end", "alignment": "symmetric"},
            "ribbed hem": {"position": "bottom", "alignment": "symmetric"},
            "zipper": {"position": "front-center", "alignment": "centered"},
            "UA logo": {"position": "chest-left", "alignment": "centered"},
            "shoulder panels": {"position": "shoulders", "alignment": "symmetric"},
            "side panels": {"position": "sides", "alignment": "symmetric"}
        }
        return placements.get(element, {"position": "unspecified", "alignment": "unspecified"})
    
    def _get_construction_details(self, style: str) -> List[Dict]:
        """Define construction details based on style category"""
        construction_details = {
            "urban athletic": [
                {"type": "flatlock seams", "purpose": "durability and comfort"},
                {"type": "reinforced stress points", "purpose": "longevity"}
            ],
            "technical performance": [
                {"type": "articulated sleeves", "purpose": "enhanced mobility"},
                {"type": "laser-cut ventilation", "purpose": "temperature regulation"}
            ],
            "minimalist design": [
                {"type": "clean finish seams", "purpose": "streamlined appearance"},
                {"type": "minimal topstitching", "purpose": "modern aesthetic"}
            ],
            "premium sportswear": [
                {"type": "bonded seams", "purpose": "premium finish"},
                {"type": "engineered fit", "purpose": "optimal drape"}
            ],
            "athleisure": [
                {"type": "comfort seams", "purpose": "all-day wearability"},
                {"type": "structured panels", "purpose": "enhanced fit"}
            ]
        }
        return construction_details.get(style, [])

    def generate_design_variations(self, base_prompt: str = "light blue hoodie") -> List[Dict]:
        """Generate multiple design variations with images"""
        logger.info(f"Generating design variations for: {base_prompt}")
        
        # Define variations based on the workflow
        variations = [
            {
                "name": "Technical Performance",
                "style": "technical and performance focused",
                "elements": ["ergonomic hood", "ventilated panels", "reflective details"],
                "emphasis": "functionality and innovation"
            },
            {
                "name": "Urban Modern",
                "style": "urban and modern",
                "elements": ["structured hood", "kangaroo pocket", "minimal branding"],
                "emphasis": "contemporary style"
            },
            {
                "name": "Premium Athletic",
                "style": "premium and sophisticated",
                "elements": ["premium zipper", "tailored fit", "refined details"],
                "emphasis": "quality and sophistication"
            },
            {
                "name": "Innovative Sport",
                "style": "innovative and functional",
                "elements": ["dynamic panels", "tech features", "performance materials"],
                "emphasis": "cutting-edge design"
            },
            {
                "name": "Minimal Essential",
                "style": "minimal and clean",
                "elements": ["clean lines", "subtle details", "refined finish"],
                "emphasis": "simplicity and elegance"
            }
        ]
        
        designs = []
        for variation in variations:
            try:
                # INPUT PHASE: Prepare design specification
                design_spec = self._create_design_spec(variation, base_prompt)
                
                # PROCESS PHASE: Generate image
                prompt = self._construct_generation_prompt(design_spec)
                logger.info(f"Generating design for {variation['name']} with prompt: {prompt}")
                
                # Generate image
                with torch.no_grad():
                    image = self._generate_image(prompt)
                
                # Save image and prompt
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = Path("results") / f"design_{variation['name'].lower().replace(' ', '_')}_{timestamp}.png"
                prompt_file = Path("results") / f"prompt_{variation['name'].lower().replace(' ', '_')}_{timestamp}.txt"
                
                # Save files
                image.save(image_path)
                with open(prompt_file, "w") as f:
                    f.write(prompt)
                
                # Analyze generated image
                analysis = self.process_design(image_path)
                
                # Add everything to design spec
                design_spec.update({
                    "prompt": prompt,
                    "prompt_file": str(prompt_file),
                    "image_path": str(image_path),
                    "analysis": analysis
                })
                designs.append(design_spec)
                
                logger.info(f"Generated and analyzed design for {variation['name']}")
                
            except Exception as e:
                logger.error(f"Error generating design for {variation['name']}: {str(e)}")
                continue
        
        return designs

    def _create_design_spec(self, variation: Dict, base_prompt: str) -> Dict:
        """Create detailed design specification"""
        return {
            "name": f"Light Blue {variation['name']} Hoodie",
            "base_prompt": base_prompt,
            "style": variation["style"],
            "elements": variation["elements"],
            "emphasis": variation["emphasis"]
        }

    def _construct_generation_prompt(self, design_spec: Dict) -> str:
        """Construct detailed prompt for image generation"""
        prompt_parts = [
            f"professional product photography of a {design_spec['base_prompt']}",
            f"style: {design_spec['style']}",
            f"featuring {', '.join(design_spec['elements'])}",
            f"emphasis on {design_spec['emphasis']}",
            "high-end fashion photography",
            "studio lighting",
            "clean background",
            "detailed fabric texture",
            "photorealistic",
            "8k resolution",
            "product showcase",
            "commercial photography"
        ]
        
        return ", ".join(prompt_parts)

    def _generate_image(self, prompt: str) -> Image.Image:
        """Generate image using Stable Diffusion"""
        output = self.image_generator(
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted, unrealistic, bad proportions",
            num_inference_steps=50,
            guidance_scale=7.5,
            height=768,
            width=768
        )
        return output.images[0]

def main():
    """Main function to generate new hoodie designs"""
    processor = DesignProcessor()
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Generate 5 design variations with prompts
    logger.info("Generating new hoodie designs...")
    designs = processor.generate_design_variations("light blue hoodie")
    
    # Save design specifications
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"hoodie_designs_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(designs, f, indent=2)
    
    # Print results
    for design in designs:
        print(f"\n{'='*50}")
        print(f"Generated Design: {design['name']}")
        print(f"Style: {design['style']}")
        print("\nDesign Elements:")
        for element in design['elements']:
            print(f"  • {element}")
        print(f"\nPrompt saved to: {design['prompt_file']}")
        print(f"{'='*50}")
    
    logger.info(f"Design generation complete. Results saved to {output_file}")

if __name__ == "__main__":
    main() 