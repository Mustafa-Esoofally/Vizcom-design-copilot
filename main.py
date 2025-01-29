import design_analysis
import design_generation
import os

def ensure_directories():
    """Ensure required directories exist"""
    for dir_name in ['Season_1', 'Season_2', 'generated']:
        os.makedirs(dir_name, exist_ok=True)

def main():
    brief = "father suiting pant blue"
    
    # Ensure directories exist
    ensure_directories()
    
    print("Phase 1: Analyzing existing designs...")
    # Run analysis phase
    design_analysis.main()
    
    print("\nPhase 2: Generating new designs...")
    # Run generation phase
    design_generation.main()
    
    print("\nWorkflow completed! Check the 'generated' directory for results.")

if __name__ == "__main__":
    main() 