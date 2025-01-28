import logging
from pathlib import Path
from process_images import InputPhaseProcessor, DesignKnowledgeGraph, DesignMemory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Initializing components...")
        # Initialize components
        knowledge_graph = DesignKnowledgeGraph()
        memory = DesignMemory(Path("cache"))
        input_processor = InputPhaseProcessor(knowledge_graph, memory)
        
        # Test brief processing
        logger.info("\nTesting brief processing...")
        brief_text = """
        Design a modern athletic jacket that combines urban architectural elements
        with technical performance features. The jacket should be suitable for
        both city wear and light athletic activities.
        """
        
        constraints = {
            "color": "blue",
            "type": "puffer jacket",
            "material": "water-resistant"
        }
        
        logger.info("Processing brief with constraints...")
        brief_analysis = input_processor.process_brief(brief_text, constraints)
        
        logger.info("\nBrief Analysis Results:")
        logger.info("Themes:")
        for theme in brief_analysis['themes']:
            logger.info(f"  - {theme['name']} (confidence: {theme['confidence']:.2f})")
            logger.info("    Related elements:")
            for elem, strength in theme['related_elements'][:3]:
                logger.info(f"      * {elem} (strength: {strength:.2f})")
        
        logger.info("\nRequirements:")
        for req in brief_analysis['requirements']:
            logger.info(f"  - {req['type']}: {req['name']} (confidence: {req['confidence']:.2f})")
        
        # Test reference image processing
        logger.info("\nTesting reference image processing...")
        # Use actual image paths from the designs directory
        reference_images = [
            Path("designs/UA+Sweatshirt+-+navy.png"),
            Path("designs/Under-Armour-PNG-Clipart.png"),
            Path("designs/png-clipart-hoodie-polar-fleece-under-armour-jacket-armor-hoodie.png")
        ]
        
        logger.info("Processing reference images...")
        for img_path in reference_images:
            logger.info(f"Checking image path: {img_path} (exists: {img_path.exists()})")
        
        reference_results = input_processor.process_reference_images(reference_images)
        
        logger.info("\nReference Image Analysis Results:")
        for idx, analysis in enumerate(reference_results['reference_analyses']):
            logger.info(f"\nImage {idx + 1}: {Path(analysis['path']).name}")
            logger.info("Design Elements:")
            for element in analysis['elements'][:5]:
                logger.info(f"  - {element['name']} (confidence: {element['confidence']:.2f})")
                logger.info("    Related elements:")
                for rel in element['related_elements'][:2]:
                    logger.info(f"      * {rel['name']} (strength: {rel['strength']:.2f})")
            
            logger.info("\nAttributes:")
            for attr in analysis['attributes'][:5]:
                logger.info(f"  - {attr['category']}: {attr['name']} (confidence: {attr['confidence']:.2f})")
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 