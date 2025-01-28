import sys
from pathlib import Path
import json

# Add the backend directory to Python path
backend_dir = Path("backend")
sys.path.append(str(backend_dir.absolute()))

print(f"Python path: {sys.path}")
print(f"Current directory: {Path.cwd()}")

try:
    from app.ml.pipeline import MLPipeline
    print("Successfully imported MLPipeline")
except Exception as e:
    print(f"Error importing MLPipeline: {str(e)}")
    sys.exit(1)

def main():
    # Initialize pipeline
    try:
        pipeline = MLPipeline()
        print("Successfully initialized MLPipeline")
    except Exception as e:
        print(f"Error initializing MLPipeline: {str(e)}")
        return
    
    # Path to test image
    image_path = Path("designs/UA+Sweatshirt+-+navy.png")
    print(f"Image path: {image_path.absolute()}")
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path.absolute()}")
        return
        
    print(f"Analyzing image: {image_path}")
    print("Starting design analysis...")
    
    try:
        # Run analysis
        results = pipeline.analyze_design(image_path)
        
        # Pretty print results
        print("\nAnalysis Results:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 