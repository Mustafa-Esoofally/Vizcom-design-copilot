import torch
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel, OwlViTProcessor, OwlViTForObjectDetection
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import logging
import networkx as nx
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DESIGNS_DIR = Path("designs")
OUTPUT_DIR = Path("results")
CACHE_DIR = Path("cache")
for dir_path in [OUTPUT_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)

@dataclass
class DesignElement:
    """Structure for design elements with relationships"""
    name: str
    category: str  # 'material', 'style', 'component'
    attributes: Dict[str, any]
    related_elements: Set[str] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "category": self.category,
            "attributes": self.attributes,
            "related_elements": list(self.related_elements) if self.related_elements else []
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DesignElement':
        data['related_elements'] = set(data.get('related_elements', []))
        return cls(**data)

class DesignKnowledgeGraph:
    """Manages relationships between design elements using a graph structure"""
    def __init__(self):
        self.graph = nx.Graph()
        self._initialize_base_elements()
        self._initialize_design_elements()
    
    def _initialize_base_elements(self):
        """Initialize base elements and their relationships from the diagram"""
        # Primary elements (from the diagram)
        primary_elements = {
            "Fire": {"category": "element", "attributes": {"type": "dynamic", "energy": "high"}},
            "Wind": {"category": "element", "attributes": {"type": "dynamic", "flow": "continuous"}},
            "Water": {"category": "element", "attributes": {"type": "fluid", "adaptability": "high"}},
            "Earth": {"category": "element", "attributes": {"type": "solid", "stability": "high"}},
            "Sky": {"category": "cosmic", "attributes": {"type": "space", "scope": "infinite"}},
            "Cloud": {"category": "state", "attributes": {"type": "transformation", "density": "variable"}},
            "Steam": {"category": "state", "attributes": {"type": "transformation", "energy": "high"}},
            "Smoke": {"category": "state", "attributes": {"type": "transformation", "visibility": "variable"}},
            "Dust": {"category": "state", "attributes": {"type": "particle", "size": "micro"}},
            "Planet": {"category": "cosmic", "attributes": {"type": "body", "scale": "macro"}}
        }
        
        # Add nodes with enhanced attributes
        for element, info in primary_elements.items():
            self.add_element(DesignElement(
                name=element,
                category=info["category"],
                attributes=info["attributes"]
            ))
        
        # Core relationships from the diagram
        relationships = [
            ("Fire", "Wind", {"type": "interaction", "strength": 0.8}),
            ("Fire", "Smoke", {"type": "production", "strength": 0.9}),
            ("Fire", "Steam", {"type": "production", "strength": 0.9}),
            ("Water", "Steam", {"type": "transformation", "strength": 0.9}),
            ("Water", "Cloud", {"type": "transformation", "strength": 0.8}),
            ("Earth", "Dust", {"type": "production", "strength": 0.7}),
            ("Wind", "Dust", {"type": "interaction", "strength": 0.6}),
            ("Wind", "Cloud", {"type": "interaction", "strength": 0.7}),
            ("Sky", "Cloud", {"type": "containment", "strength": 0.8}),
            ("Sky", "Planet", {"type": "containment", "strength": 0.9})
        ]
        
        for elem1, elem2, attrs in relationships:
            self.add_relationship(elem1, elem2, weight=attrs["strength"], attributes=attrs)
    
    def _initialize_design_elements(self):
        """Initialize design-specific elements and their relationships"""
        design_elements = {
            "Material": {"category": "design", "attributes": {"type": "physical", "importance": "high"}},
            "Form": {"category": "design", "attributes": {"type": "structural", "importance": "high"}},
            "Function": {"category": "design", "attributes": {"type": "utility", "importance": "high"}},
            "Aesthetic": {"category": "design", "attributes": {"type": "visual", "importance": "high"}},
            "Performance": {"category": "design", "attributes": {"type": "functional", "importance": "high"}}
        }
        
        # Add design nodes
        for element, info in design_elements.items():
            self.add_element(DesignElement(
                name=element,
                category=info["category"],
                attributes=info["attributes"]
            ))
        
        # Design relationships
        design_relationships = [
            ("Material", "Form", {"type": "influence", "strength": 0.9}),
            ("Form", "Function", {"type": "enablement", "strength": 0.8}),
            ("Function", "Performance", {"type": "impact", "strength": 0.9}),
            ("Form", "Aesthetic", {"type": "expression", "strength": 0.8}),
            ("Material", "Performance", {"type": "contribution", "strength": 0.7})
        ]
        
        for elem1, elem2, attrs in design_relationships:
            self.add_relationship(elem1, elem2, weight=attrs["strength"], attributes=attrs)
        
        # Connect to base elements
        element_connections = [
            ("Material", "Earth", {"type": "derivation", "strength": 0.6}),
            ("Form", "Wind", {"type": "inspiration", "strength": 0.5}),
            ("Function", "Water", {"type": "adaptation", "strength": 0.5}),
            ("Performance", "Fire", {"type": "energy", "strength": 0.6}),
            ("Aesthetic", "Sky", {"type": "inspiration", "strength": 0.5})
        ]
        
        for elem1, elem2, attrs in element_connections:
            self.add_relationship(elem1, elem2, weight=attrs["strength"], attributes=attrs)
    
    def add_element(self, element: DesignElement):
        """Add a design element to the graph"""
        self.graph.add_node(
            element.name,
            category=element.category,
            attributes=element.attributes
        )
    
    def add_relationship(self, elem1: str, elem2: str, weight: float = 1.0, attributes: Dict = None):
        """Add or update a relationship between elements with attributes"""
        if elem1 in self.graph and elem2 in self.graph:
            self.graph.add_edge(
                elem1, elem2,
                weight=weight,
                **attributes if attributes else {}
            )
    
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
                        # Calculate relationship strength based on distance
                        strength = 1.0 / (1.0 + distance)
                        related.append((node, strength))
                except nx.NetworkXNoPath:
                    continue
        
        return sorted(related, key=lambda x: x[1], reverse=True)
    
    def find_design_path(self, start: str, end: str) -> List[str]:
        """Find a path between two design elements"""
        try:
            return nx.shortest_path(self.graph, start, end, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_element_info(self, element: str) -> Optional[Dict]:
        """Get detailed information about an element"""
        if element in self.graph:
            return {
                "category": self.graph.nodes[element]["category"],
                "attributes": self.graph.nodes[element]["attributes"],
                "connections": list(self.graph.neighbors(element))
            }
        return None

    def visualize(self, output_path: str = "knowledge_graph.png"):
        """Visualize the knowledge graph"""
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Draw nodes with different colors based on category
        categories = nx.get_node_attributes(self.graph, 'category')
        colors = {'element': 'lightblue', 'state': 'lightgreen', 
                 'cosmic': 'lightpink', 'design': 'lightyellow'}
        
        for category, color in colors.items():
            nodes = [node for node, cat in categories.items() if cat == category]
            nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes, 
                                 node_color=color, node_size=2000)
        
        # Draw edges with varying thickness based on weight
        weights = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edges(self.graph, pos, width=[w*2 for w in weights.values()])
        
        # Add labels
        nx.draw_networkx_labels(self.graph, pos)
        
        plt.title("Design Knowledge Graph")
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()

@dataclass
class DesignBrief:
    """Structure for design brief analysis"""
    theme: str
    style_attributes: List[str]
    constraints: List[str]
    target_elements: List[str]
    image_specific_elements: List[Tuple[str, float]]

@dataclass
class StyleAnalysis:
    """Structure for style analysis results"""
    style_vector: np.ndarray
    dominant_elements: List[Tuple[str, float]]
    color_palette: List[str]
    design_score: float

@dataclass
class MemoryItem:
    """Structure for memory storage with enhanced state management"""
    timestamp: datetime
    content_type: str  # 'style', 'brief', 'element'
    data: Dict
    source: str
    relevance_score: float
    status: str = "active"  # active, archived, or deleted
    tags: List[str] = None
    priority: int = 1  # 1-5, with 5 being highest priority
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
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
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class DesignMemory:
    """Enhanced memory management for design processing"""
    def __init__(self, cache_dir: Path):
        self.short_term = []  # Project-specific memory
        self.long_term = []   # Brand guidelines and persistent memory
        self.cache_dir = cache_dir
        self.memory_file = self.cache_dir / "design_memory.json"
        self._load_persistent_memory()
    
    def _load_persistent_memory(self):
        """Load persistent memory from cache with error handling"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.long_term = [MemoryItem.from_dict(item) for item in data.get('long_term', [])]
                    logger.info(f"Loaded {len(self.long_term)} items from persistent memory")
        except Exception as e:
            logger.error(f"Error loading persistent memory: {e}")
            self.long_term = []
    
    def add_memory(self, content_type: str, data: Dict, source: str, 
                  is_persistent: bool = False, relevance_score: float = 1.0,
                  tags: List[str] = None, priority: int = 1):
        """Add new memory item with enhanced metadata"""
        memory_item = MemoryItem(
            timestamp=datetime.now(),
            content_type=content_type,
            data=data,
            source=source,
            relevance_score=relevance_score,
            tags=tags or [],
            priority=priority
        )
        
        if is_persistent:
            self.long_term.append(memory_item)
            self._save_persistent_memory()
        else:
            self.short_term.append(memory_item)
            # Cleanup old short-term memory items
            self._cleanup_short_term()
    
    def query_memory(self, content_type: str, query_vector: np.ndarray = None, 
                    top_k: int = 5, min_relevance: float = 0.5,
                    tags: List[str] = None) -> List[MemoryItem]:
        """Enhanced memory query with filtering"""
        memory_items = self.short_term + self.long_term
        
        # Filter by content type and status
        relevant_items = [
            item for item in memory_items 
            if item.content_type == content_type 
            and item.status == "active"
            and item.relevance_score >= min_relevance
        ]
        
        # Filter by tags if provided
        if tags:
            relevant_items = [
                item for item in relevant_items
                if item.tags and any(tag in item.tags for tag in tags)
            ]
        
        # Calculate similarities if query vector provided
        if query_vector is not None:
            similarities = []
            for item in relevant_items:
                if 'vector' in item.data:
                    similarity = float(np.dot(query_vector, item.data['vector']))
                    # Weight by priority and recency
                    recency_weight = 1.0 / (1.0 + (datetime.now() - item.timestamp).days)
                    final_score = similarity * item.priority * recency_weight
                    similarities.append((final_score, item))
            
            # Sort by final score and return top-k
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [item for _, item in similarities[:top_k]]
        
        # If no query vector, sort by recency and priority
        relevant_items.sort(
            key=lambda x: (x.priority, x.timestamp),
            reverse=True
        )
        return relevant_items[:top_k]
    
    def _cleanup_short_term(self, max_age_days: int = 7, max_items: int = 1000):
        """Cleanup old short-term memory items"""
        now = datetime.now()
        self.short_term = [
            item for item in self.short_term
            if (now - item.timestamp).days <= max_age_days
        ]
        
        if len(self.short_term) > max_items:
            # Keep most recent and highest priority items
            self.short_term.sort(
                key=lambda x: (x.priority, x.timestamp), 
                reverse=True
            )
            self.short_term = self.short_term[:max_items]
    
    def _save_persistent_memory(self):
        """Save persistent memory to cache with error handling"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump({
                    'long_term': [item.to_dict() for item in self.long_term]
                }, f, indent=2)
            logger.info(f"Saved {len(self.long_term)} items to persistent memory")
        except Exception as e:
            logger.error(f"Error saving persistent memory: {e}")

class InputPhaseProcessor:
    """Handles the input phase of the design process"""
    def __init__(self, knowledge_graph: DesignKnowledgeGraph, memory: DesignMemory):
        self.knowledge_graph = knowledge_graph
        self.memory = memory
        self.clip_model, self.clip_processor = self._load_clip()
    
    def _load_clip(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """Load CLIP model and processor"""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

    def process_brief(self, brief_text: str, constraints: Dict[str, any]) -> Dict:
        """Process design brief and constraints"""
        # Analyze text brief
        text_features = self._extract_text_features(brief_text)
        
        # Process constraints
        processed_constraints = self._process_constraints(constraints)
        
        # Query memory for similar briefs
        similar_briefs = self.memory.query_memory(
            'brief',
            query_vector=text_features.detach().numpy(),
            tags=['design_brief']
        )
        
        # Extract key themes and requirements
        themes = self._extract_themes(brief_text, text_features)
        requirements = self._extract_requirements(brief_text, processed_constraints)
        
        brief_analysis = {
            'text': brief_text,
            'themes': themes,
            'requirements': requirements,
            'constraints': processed_constraints,
            'similar_briefs': [brief.data for brief in similar_briefs],
            'vector': text_features.detach().numpy().tolist()  # Convert to list for JSON serialization
        }
        
        # Store in memory
        self.memory.add_memory(
            content_type='brief',
            data=brief_analysis,
            source='user_input',
            is_persistent=True,
            tags=['design_brief'],
            priority=5
        )
        
        return brief_analysis
    
    def process_reference_images(self, image_paths: List[Path]) -> Dict:
        """Process reference images"""
        reference_analyses = []
        
        for img_path in image_paths:
        # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            
            # Extract image features
            image_inputs = self.clip_processor(
                images=image,
                return_tensors="pt"
            )
            image_features = self.clip_model.get_image_features(**image_inputs)
            
            # Query memory for similar styles
            similar_styles = self.memory.query_memory(
                'style',
                query_vector=image_features[0].detach().numpy().tolist(),  # Convert to list for JSON
                tags=['reference_style']
            )
            
            # Extract design elements and attributes
            elements = self._extract_design_elements(image, img_path)
            attributes = self._extract_design_attributes(image_features[0])  # Pass single tensor
            
            analysis = {
                'path': str(img_path),
                'elements': elements,
                'attributes': attributes,
                'similar_styles': [style.data for style in similar_styles],
                'vector': image_features[0].detach().numpy().tolist()  # Convert to list for JSON
            }
            
            reference_analyses.append(analysis)
            
            # Store in memory
            self.memory.add_memory(
                content_type='reference',
                data=analysis,
                source=str(img_path),
                is_persistent=True,
                tags=['reference_style'],
                priority=4
            )
        
        return {'reference_analyses': reference_analyses}
    
    def _extract_text_features(self, text: str) -> torch.Tensor:
        """Extract features from text using CLIP"""
        text_inputs = self.clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True
        )
        return self.clip_model.get_text_features(**text_inputs)[0]
    
    def _process_constraints(self, constraints: Dict[str, any]) -> Dict:
        """Process and validate design constraints"""
        processed = {}
        
        for constraint_type, value in constraints.items():
            # Get related elements from knowledge graph
            element_info = self.knowledge_graph.get_element_info(str(value))
            related_elements = []
            
            if element_info:
                related = self.knowledge_graph.get_related_elements(str(value))
                related_elements = [
                    {
                        'name': rel_name,
                        'strength': rel_strength,
                        'info': self.knowledge_graph.get_element_info(rel_name)
                    }
                    for rel_name, rel_strength in related
                ]
            
            processed[constraint_type] = {
                'value': value,
                'element_info': element_info,
                'related_elements': related_elements
            }
        
        return processed
    
    def _extract_themes(self, text: str, text_features: torch.Tensor) -> List[Dict]:
        """Extract design themes from text"""
        themes = []
        
        # Define theme categories
        theme_categories = [
            "urban architecture", "modern minimalist", "technical sportswear",
            "performance athletic", "sleek design", "functional outerwear",
            "innovative construction", "premium athletic"
        ]
        
        # Get theme predictions
        theme_inputs = self.clip_processor(
            text=theme_categories,
            return_tensors="pt",
            padding=True
        )
        theme_features = self.clip_model.get_text_features(**theme_inputs)
        
        similarities = torch.nn.functional.cosine_similarity(
            text_features.unsqueeze(0),
            theme_features,
            dim=1
        )
        
        # Extract relevant themes
        for theme, score in zip(theme_categories, similarities):
            if score > 0.3:
                themes.append({
                    'name': theme,
                    'confidence': float(score),
                    'related_elements': [
                        (elem, strength) 
                        for elem, strength in self.knowledge_graph.get_related_elements(theme)
                    ]
                })
        
        return sorted(themes, key=lambda x: x['confidence'], reverse=True)
    
    def _extract_requirements(self, text: str, constraints: Dict) -> List[Dict]:
        """Extract design requirements from text and constraints"""
        requirements = []
        
        # Extract material requirements
        material_properties = [
            "water-resistant", "breathable", "lightweight",
            "durable", "stretchy", "insulated", "ventilated"
        ]
        
        # Get requirement predictions
        req_inputs = self.clip_processor(
            text=[text] + material_properties,
            return_tensors="pt",
            padding=True
        )
        req_features = self.clip_model.get_text_features(**req_inputs)
        
        similarities = torch.nn.functional.cosine_similarity(
            req_features[0].unsqueeze(0),
            req_features[1:],
            dim=1
        )
        
        # Extract relevant requirements
        for prop, score in zip(material_properties, similarities):
            if score > 0.3:
                requirements.append({
                    'type': 'material',
                    'name': prop,
                    'confidence': float(score),
                    'source': 'text_analysis'
                })
        
        # Add requirements from constraints
        for constraint_type, data in constraints.items():
            requirements.append({
                'type': constraint_type,
                'name': data['value'],
                'confidence': 1.0,
                'source': 'constraints'
            })
        
        return requirements
    
    def _extract_design_elements(self, image: Image, image_path: Path) -> List[Dict]:
        """Extract design elements from reference image"""
        # Define target elements
        target_elements = [
            "collar", "zipper", "pocket", "seam", "logo",
            "sleeve", "hood", "cuff", "hem", "panel"
        ]
        
        # Process image and text
        inputs = self.clip_processor(
            images=image,
            text=target_elements,
            return_tensors="pt",
            padding=True
        )
        
        outputs = self.clip_model(**inputs)
        logits = outputs.logits_per_image[0]  # Get logits for single image
        probs = torch.nn.functional.softmax(logits, dim=0)  # Apply softmax on correct dimension
        
        # Extract elements with high confidence
        elements = []
        for element, prob in zip(target_elements, probs):
            prob_val = float(prob.item())  # Convert to float scalar
            if prob_val > 0.3:
                # Get related elements from knowledge graph
                related = self.knowledge_graph.get_related_elements(element)
                elements.append({
                    'name': element,
                    'confidence': prob_val,
                    'related_elements': [
                        {'name': rel_name, 'strength': rel_strength}
                        for rel_name, rel_strength in related
                    ]
                })
        
        return sorted(elements, key=lambda x: x['confidence'], reverse=True)
    
    def _extract_design_attributes(self, image_features: torch.Tensor) -> List[Dict]:
        """Extract design attributes from image features"""
        # Define attribute categories
        attribute_categories = {
            'style': ["modern", "minimal", "technical", "sporty", "sleek"],
            'form': ["fitted", "loose", "structured", "layered", "seamless"],
            'detail': ["geometric", "textured", "patterned", "contrast", "tonal"]
        }
        
        attributes = []
        for category, attrs in attribute_categories.items():
            # Get attribute predictions
            attr_inputs = self.clip_processor(
                text=attrs,
                return_tensors="pt",
                padding=True
            )
            attr_features = self.clip_model.get_text_features(**attr_inputs)
            
            # Reshape image features to match dimensions
            image_features_2d = image_features.unsqueeze(0)  # Add batch dimension
            
            # Normalize features
            image_features_norm = image_features_2d / image_features_2d.norm(dim=-1, keepdim=True)
            attr_features_norm = attr_features / attr_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            similarity = torch.nn.functional.cosine_similarity(
                image_features_norm.unsqueeze(1).expand(-1, attr_features.size(0), -1),
                attr_features_norm.unsqueeze(0),
                dim=2
            )
            scores = similarity[0]  # Get scores for the first (and only) image
            
            # Extract relevant attributes
            for attr, score in zip(attrs, scores):
                score_val = float(score.item())  # Convert to float scalar
                if score_val > 0.3:
                    attributes.append({
                        'category': category,
                        'name': attr,
                        'confidence': score_val
                    })
        
        return sorted(attributes, key=lambda x: x['confidence'], reverse=True)

class DesignProcessor:
    def __init__(self):
        """Initialize models and processors"""
        logger.info("Loading models...")
        # CLIP for style understanding
        self.clip_model, self.clip_processor = self._load_clip()
        # OwlViT for element localization
        self.owlvit_model, self.owlvit_processor = self._load_owlvit()
        
        # Initialize memory system
        self.memory = DesignMemory(CACHE_DIR)
        
        # Initialize knowledge graph
        self.knowledge_graph = DesignKnowledgeGraph()
        
        # Style categories with relationships
        self.style_categories = {
            "urban architecture": ["modern", "geometric", "structural"],
            "modern minimalist": ["clean", "simple", "refined"],
            "technical sportswear": ["functional", "performance", "innovative"],
            "performance athletic": ["dynamic", "ergonomic", "efficient"],
            "sleek design": ["streamlined", "minimal", "sophisticated"]
        }
        
        # Design elements with material relationships
        self.design_elements = {
            "collar": ["fabric", "structure"],
            "zipper": ["metal", "plastic"],
            "pocket": ["fabric", "structure"],
            "seam": ["construction", "strength"],
            "logo placement": ["branding", "visibility"]
        }
        
        # Material properties with relationships
        self.material_properties = {
            "water-resistant": ["protection", "durability"],
            "breathable": ["comfort", "performance"],
            "lightweight": ["mobility", "comfort"],
            "durable": ["strength", "longevity"],
            "stretchy": ["flexibility", "comfort"]
        }

    def _load_clip(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """Load CLIP model and processor"""
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor

    def _load_owlvit(self) -> Tuple[OwlViTForObjectDetection, OwlViTProcessor]:
        """Load OwlViT model and processor"""
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        return model, processor

    def process_input_phase(self, brief_text: str, reference_images: List[Path], 
                          constraints: Dict[str, any]) -> Dict:
        """Process input phase with enhanced context understanding"""
        logger.info("Processing input phase...")
        
        # Analyze brief with knowledge graph context
        brief_analysis = self.analyze_brief(brief_text)
        
        # Process constraints
        processed_constraints = self._process_constraints(constraints)
        
        # Analyze reference images
        reference_analyses = []
        for img_path in reference_images:
            analysis = self.analyze_style(img_path)
            # Enhance analysis with knowledge graph relationships
            for element, score in analysis.dominant_elements:
                related = self.knowledge_graph.get_related_elements(element)
                for rel_elem, rel_strength in related:
                    analysis.dominant_elements.append(
                        (rel_elem, score * rel_strength)
                    )
            reference_analyses.append(analysis)
        
        # Combine analyses
        input_phase_result = {
            "brief_analysis": asdict(brief_analysis),
            "constraints": processed_constraints,
            "reference_analyses": [asdict(analysis) for analysis in reference_analyses],
            "knowledge_paths": self._extract_relevant_paths(brief_analysis)
        }
        
        # Store in memory
        self.memory.add_memory(
            content_type='input_phase',
            data=input_phase_result,
            source='user_input',
            is_persistent=True,
            tags=['input', 'brief', 'reference'],
            priority=5
        )
        
        return input_phase_result

    def _process_constraints(self, constraints: Dict[str, any]) -> Dict:
        """Process and validate design constraints"""
        processed = {}
        
        for constraint_type, value in constraints.items():
            # Validate against knowledge graph
            if constraint_type in ["material", "style", "component"]:
                element_info = self.knowledge_graph.get_element_info(value)
                if element_info:
                    processed[constraint_type] = {
                        "value": value,
                        "info": element_info,
                        "related_elements": self.knowledge_graph.get_related_elements(value)
                    }
            else:
                processed[constraint_type] = {"value": value}
        
        return processed

    def _extract_relevant_paths(self, brief: DesignBrief) -> List[List[str]]:
        """Extract relevant design paths from knowledge graph based on brief"""
        paths = []
        
        # Extract key elements from brief
        key_elements = set()
        for attr in brief.style_attributes:
            element = attr.split(" (")[0]  # Remove confidence score
            if self.knowledge_graph.get_element_info(element):
                key_elements.add(element)
        
        # Find paths between key elements
        for start in key_elements:
            for end in key_elements:
                if start != end:
                    path = self.knowledge_graph.find_design_path(start, end)
                    if path:
                        paths.append(path)
        
        return paths

    def analyze_brief(self, brief_text: str, image_path: Optional[Path] = None) -> DesignBrief:
        """Analyze design brief with enhanced knowledge graph integration"""
        # Get base analysis
        brief = self._analyze_brief_base(brief_text, image_path)
        
        # Enhance with knowledge graph relationships
        enhanced_attributes = []
        for attr in brief.style_attributes:
            element = attr.split(" (")[0]
            score = float(attr.split("(")[1].rstrip(")"))
            
            # Get related elements from knowledge graph
            related = self.knowledge_graph.get_related_elements(element)
            enhanced_attributes.append(f"{element} ({score:.2f})")
            
            # Add related elements with adjusted scores
            for rel_elem, rel_strength in related:
                enhanced_score = score * rel_strength
                if enhanced_score > 0.3:  # Threshold for related elements
                    enhanced_attributes.append(f"{rel_elem} ({enhanced_score:.2f})")
        
        brief.style_attributes = enhanced_attributes
        return brief

    def _analyze_brief_base(self, brief_text: str, image_path: Optional[Path] = None) -> DesignBrief:
        """Base implementation of brief analysis"""
        # Query memory for similar briefs
        similar_briefs = self.memory.query_memory('brief', tags=['urban', 'architecture'])
        
        # Enhanced style categories with more detail
        detailed_style_categories = [
            "urban architecture", "modern minimalist", "technical sportswear",
            "performance athletic", "sleek design", "functional outerwear",
            "innovative construction", "premium athletic", "street-inspired",
            "architectural details", "geometric patterns", "structured silhouettes"
        ]
        
        # Add style categories from memory
        if similar_briefs:
            for brief in similar_briefs:
                if 'style_categories' in brief.data:
                    detailed_style_categories.extend(brief.data['style_categories'])
            detailed_style_categories = list(set(detailed_style_categories))
        
        # Enhanced design elements with specific details
        detailed_design_elements = [
            "raglan sleeve construction", "ergonomic seam placement",
            "ventilation panels", "waterproof zippers", "reflective trim",
            "adjustable hood", "reinforced stress points", "hidden pockets",
            "articulated elbows", "storm flap", "mesh lining",
            "elastic cuffs", "drawcord hem", "logo placement",
            "bonded seams", "laser-cut vents"
        ]
        
        # Material and performance features
        material_features = [
            "water-resistant shell", "breathable membrane",
            "moisture-wicking fabric", "4-way stretch material",
            "abrasion-resistant panels", "lightweight insulation",
            "quick-dry technology", "thermal regulation",
            "wind-resistant layer", "durable water repellent"
        ]
        
        # Use CLIP for detailed analysis
        text_inputs = self.clip_processor(
            text=[brief_text] + detailed_style_categories + detailed_design_elements + material_features,
            return_tensors="pt",
            padding=True
        )
        text_features = self.clip_model.get_text_features(**text_inputs)
        
        # Get detailed style relevance scores
        style_scores = torch.nn.functional.cosine_similarity(
            text_features[0:1], 
            text_features[1:len(detailed_style_categories)+1], 
            dim=1
        )
        
        # Get design element scores
        element_scores = torch.nn.functional.cosine_similarity(
            text_features[0:1],
            text_features[len(detailed_style_categories)+1:len(detailed_style_categories)+len(detailed_design_elements)+1],
            dim=1
        )
        
        # Get material feature scores
        material_scores = torch.nn.functional.cosine_similarity(
            text_features[0:1],
            text_features[-len(material_features):],
            dim=1
        )
        
        # Extract relevant attributes with scores
        relevant_styles = [
            (style, float(score)) 
            for style, score in zip(detailed_style_categories, style_scores)
            if score > 0.3
        ]
        
        relevant_elements = [
            (element, float(score))
            for element, score in zip(detailed_design_elements, element_scores)
            if score > 0.3
        ]
        
        relevant_materials = [
            (material, float(score))
            for material, score in zip(material_features, material_scores)
            if score > 0.3
        ]
        
        # If image is provided, analyze image-specific elements
        image_specific_elements = []
        if image_path:
            image = Image.open(image_path).convert('RGB')
            image_inputs = self.clip_processor(
                images=image,
                text=detailed_design_elements + material_features,
                return_tensors="pt",
                padding=True
            )
            
            outputs = self.clip_model(**image_inputs)
            probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=1)
            
            image_specific_elements = [
                (element, float(prob))
                for element, prob in zip(detailed_design_elements + material_features, probs[0].tolist())
                if prob > 0.3
            ]
        
        brief = DesignBrief(
            theme="urban architecture",
            style_attributes=[f"{style} ({score:.2f})" for style, score in relevant_styles],
            constraints=[f"{material} ({score:.2f})" for material, score in relevant_materials],
            target_elements=[f"{element} ({score:.2f})" for element, score in relevant_elements],
            image_specific_elements=image_specific_elements if image_path else []
        )
        
        # Store brief in memory
        self.memory.add_memory(
            content_type='brief',
            data={
                'text': brief_text,
                'style_categories': detailed_style_categories,
                'analysis': asdict(brief)
            },
            source='user_input',
            is_persistent=True,
            tags=['urban', 'architecture'],
            priority=3
        )
        
        return brief

    def analyze_style(self, image_path: Path) -> StyleAnalysis:
        """Analyze design style using CLIP and memory"""
        image = Image.open(image_path).convert('RGB')
        
        # Prepare text inputs as flat lists of strings
        style_texts = []
        for style, attributes in self.style_categories.items():
            style_texts.append(str(style))
            style_texts.extend([str(attr) for attr in attributes])
        
        design_texts = []
        for element, attributes in self.design_elements.items():
            design_texts.append(str(element))
            design_texts.extend([str(attr) for attr in attributes])
        
        material_texts = []
        for property_name, attributes in self.material_properties.items():
            material_texts.append(str(property_name))
            material_texts.extend([str(attr) for attr in attributes])
        
        all_texts = [text for text in (style_texts + design_texts + material_texts) if text.strip()]
        
        # Get style embeddings
        inputs = self.clip_processor(
            images=image,
            text=all_texts,
            return_tensors="pt",
            padding=True
        )
        
        outputs = self.clip_model(**inputs)
        image_features = self.clip_model.get_image_features(inputs['pixel_values'])
        
        # Get element predictions
        probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=1)
        elements = [(text, float(prob)) for text, prob in 
                   zip(all_texts, probs[0].tolist())]
        elements.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate design score (simplified)
        design_score = float(torch.mean(probs[0]))
        
        # Query similar styles from memory
        similar_styles = self.memory.query_memory(
            'style',
            query_vector=image_features[0].detach().numpy(),
            top_k=3
        )
        
        # Enhance analysis with memory context
        if similar_styles:
            # Combine current analysis with historical insights
            combined_elements = elements.copy()
            for memory_item in similar_styles:
                if 'dominant_elements' in memory_item.data:
                    for elem, score in memory_item.data['dominant_elements']:
                        if elem not in [e[0] for e in combined_elements]:
                            combined_elements.append((str(elem), float(score) * 0.8))  # Discount historical scores
            
            # Update elements list
            combined_elements.sort(key=lambda x: x[1], reverse=True)
            elements = combined_elements[:10]  # Keep top 10 elements
        
        style_analysis = StyleAnalysis(
            style_vector=image_features[0].detach().numpy(),
            dominant_elements=elements,
            color_palette=["#000000"],  # Placeholder for color extraction
            design_score=design_score
        )
        
        # Store style analysis in memory
        self.memory.add_memory(
            content_type='style',
            data={
                'vector': style_analysis.style_vector,
                'dominant_elements': [(str(elem), float(score)) for elem, score in style_analysis.dominant_elements],
                'design_score': float(style_analysis.design_score)
            },
            source=str(image_path),
            tags=['style_analysis'],
            priority=2
        )
        
        return style_analysis

    def localize_elements(self, image_path: Path) -> Dict[str, List[float]]:
        """Localize design elements using OwlViT"""
        image = Image.open(image_path).convert('RGB')
        
        # Prepare text queries from design elements
        text_queries = list(self.design_elements.keys())
        
        # Process image and text with OwlViT
        inputs = self.owlvit_processor(
            images=image,
            text=text_queries,
            return_tensors="pt"
        )
        
        outputs = self.owlvit_model(**inputs)
        
        # Process predictions
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.owlvit_processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.1
        )[0]
        
        elements = {}
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            if score > 0.5:  # Confidence threshold
                element = text_queries[label]
                elements[element] = {
                    "box": box.tolist(),
                    "score": float(score),
                    "attributes": self.design_elements[element]
                }
        
        return elements

    def process_design(self, image_path: Path, brief: Optional[DesignBrief] = None) -> Dict:
        """Process a single design image with memory integration"""
        try:
            logger.info(f"Processing {image_path.name}...")
            
            # Analyze style
            style_analysis = self.analyze_style(image_path)
            
            # Localize elements
            elements = self.localize_elements(image_path)
            
            # Get related elements from knowledge graph
            enhanced_elements = {}
            for element_name, box in elements.items():
                related = self.knowledge_graph.get_related_elements(element_name)
                enhanced_elements[element_name] = {
                    "box": box,
                    "related_elements": [
                        {"name": rel_name, "strength": rel_strength}
                        for rel_name, rel_strength in related
                    ]
                }
            
            # Compile results
            results = {
                "filename": image_path.name,
                "style_analysis": {
                    "dominant_elements": [
                        {"element": elem, "confidence": float(conf)}
                        for elem, conf in style_analysis.dominant_elements
                    ],
                    "design_score": style_analysis.design_score
                },
                "localized_elements": enhanced_elements,
                "timestamp": datetime.now().isoformat()
            }
            
            if brief:
                results["detailed_brief_analysis"] = {
                    "theme": brief.theme,
                    "style_attributes": brief.style_attributes,
                    "constraints": brief.constraints,
                    "target_elements": brief.target_elements,
                    "image_specific_elements": [
                        {"element": elem, "confidence": float(conf)}
                        for elem, conf in brief.image_specific_elements
                    ] if brief.image_specific_elements else []
                }
            
            return results
    
    except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
        return {"error": str(e)}

def main():
    """Main function to process all design images"""
    processor = DesignProcessor()
    designs_dir = Path("designs")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Example brief text
    brief_text = """Design a pre-season Outerwear collection inspired by urban architecture. 
    Focus on technical performance, weather protection, and modern aesthetic. 
    Incorporate innovative construction methods and premium materials."""
    
    results = []
    
    # Process each image
    for image_path in designs_dir.glob("*.png"):
        logger.info(f"\nAnalyzing {image_path.name}...")
        
        # Get detailed brief analysis for this image
        brief_analysis = processor.analyze_brief(brief_text, image_path)
        
        # Process design
        design_results = processor.process_design(image_path, brief_analysis)
        
        # Add detailed brief analysis to results
        design_results["detailed_brief_analysis"] = {
            "theme": brief_analysis.theme,
            "style_attributes": brief_analysis.style_attributes,
            "constraints": brief_analysis.constraints,
            "target_elements": brief_analysis.target_elements,
            "image_specific_elements": brief_analysis.image_specific_elements
        }
        
        results.append(design_results)
        
        # Print detailed analysis for this image
        print(f"\nDetailed Analysis for {image_path.name}:")
        print("Style Attributes:")
        for style in brief_analysis.style_attributes:
            print(f"  - {style}")
        print("\nDesign Elements:")
        for element in brief_analysis.target_elements:
            print(f"  - {element}")
        print("\nMaterial Constraints:")
        for constraint in brief_analysis.constraints:
            print(f"  - {constraint}")
        if brief_analysis.image_specific_elements:
            print("\nImage-Specific Elements:")
            for element, score in brief_analysis.image_specific_elements:
                print(f"  - {element} ({score:.2f})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"design_analysis_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nAnalysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    main() 