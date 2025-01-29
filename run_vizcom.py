import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'fashion-clip'))
from pathlib import Path
import logging
from vizcom_system import VizcomSystem


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize paths
    season1_dir = Path("Season_1")
    season2_dir = Path("Season_2")
    
    # Initialize system
    system = VizcomSystem()
    
    # Process input designs with brief
    text_brief = "father suiting pant blue"
    system.process_input_designs(
        season_dirs=[season1_dir, season2_dir],
        text_brief=text_brief
    )
    
    # Generate design paths
    paths = system.generate_design_paths(num_paths=5)
    logger.info("Generated design paths:")
    for i, path in enumerate(paths, 1):
        logger.info(f"\nPath {i}:")
        logger.info(f"Base: {path['base_type']}")
        logger.info(f"Elements: {', '.join(path['elements'])}")
        logger.info(f"Description: {path['description']}")
    
    # Select first path and generate designs
    if paths:
        selected_path = paths[0]
        logger.info(f"\nGenerating designs for path: {selected_path['description']}")
        images = system.generate_designs(selected_path, num_variations=5)
        
        # Save generated images
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        for i, image in enumerate(images, 1):
            image_path = results_dir / f"design_variation_{i}.png"
            image.save(image_path)
            logger.info(f"Saved design variation {i} to {image_path}")
    else:
        logger.warning("No valid design paths generated")

if __name__ == "__main__":
    main() 