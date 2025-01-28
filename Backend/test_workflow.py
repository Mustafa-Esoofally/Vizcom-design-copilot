import logging
from pathlib import Path
from process_images import DesignProcessor, DesignKnowledgeGraph, DesignMemory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize components
        logger.info("Initializing components...")
        knowledge_graph = DesignKnowledgeGraph()
        memory = DesignMemory(Path("cache"))
        processor = DesignProcessor()
        
        # Example brief text
        brief_text = """Design a versatile athletic apparel collection that combines 
        performance features with modern aesthetics. Focus on moisture management, 
        comfort, and brand consistency across different garment types."""
        
        # Process each design image
        design_files = [
            "png-clipart-hoodie-polar-fleece-under-armour-jacket-armor-hoodie.png",
            "png-transparent-t-shirt-clothing-sleeve-under-armour-t-shirt.png",
            "TEEUAPERFCOTTONUWWWOVWARUALOGOl.png",
            "UA+Sweatshirt+-+navy.png",
            "Under-Armour-PNG-Clipart.png"
        ]
        
        results = []
        designs_dir = Path("designs")
        
        logger.info("\nProcessing design images...")
        for filename in design_files:
            image_path = designs_dir / filename
            if image_path.exists():
                logger.info(f"\nAnalyzing {filename}...")
                
                # Process design
                design_results = processor.process_design(image_path)
                results.append(design_results)
                
                # Print analysis summary
                if "error" not in design_results:
                    print(f"\nAnalysis for {filename}:")
                    print("Style Elements:")
                    for elem in design_results["style_analysis"]["dominant_elements"]:
                        print(f"  - {elem['element']} (confidence: {elem['confidence']:.2f})")
                    
                    print("\nLocalized Elements:")
                    for elem_name, elem_data in design_results["localized_elements"].items():
                        print(f"  - {elem_name}")
                        if "related_elements" in elem_data:
                            print("    Related elements:")
                            for rel in elem_data["related_elements"][:3]:
                                print(f"      * {rel['name']} (strength: {rel['strength']:.2f})")
                else:
                    logger.error(f"Error processing {filename}: {design_results['error']}")
            else:
                logger.warning(f"File not found: {image_path}")
        
        logger.info("\nAnalysis complete!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 