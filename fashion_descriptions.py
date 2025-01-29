import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import base64
from tqdm import tqdm

# Load environment variables
load_dotenv()

def encode_image(image_path: str) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_fashion_description(image_path: str, model: ChatOpenAI) -> str:
    """Generate a detailed fashion description using GPT-4V."""
    base64_image = encode_image(image_path)
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
   - Seasonal relevance

Please format the response with clear sections and detailed observations."""
                }
            ]
        )
    ]
    response = model.invoke(messages)
    return response.content

def main():
    # Initialize GPT-4V
    model = ChatOpenAI(model="chatgpt-4o-latest", max_tokens=1000)
    
    # Create descriptions directory
    os.makedirs("descriptions", exist_ok=True)
    
    # Process Season 1
    print("\nAnalyzing Season 1 Designs:")
    print("-" * 80)
    season1_path = "Vizcom/Season_1"
    for image_file in sorted(os.listdir(season1_path)):
        if image_file.endswith(('.webp', '.jpg', '.jpeg', '.png')):
            name = Path(image_file).stem
            desc_file = f"descriptions/{name}.txt"
            
            if not os.path.exists(desc_file):
                image_path = os.path.join(season1_path, image_file)
                print(f"\nGenerating description for: {image_file}")
                description = generate_fashion_description(image_path, model)
                
                # Save description to file
                with open(desc_file, 'w', encoding='utf-8') as f:
                    f.write(description)
                print(f"Description saved to {desc_file}")
            else:
                print(f"Description already exists for {image_file}")
    
    # Process Season 2
    print("\nAnalyzing Season 2 Designs:")
    print("-" * 80)
    season2_path = "Vizcom/Season_2"
    for image_file in sorted(os.listdir(season2_path)):
        if image_file.endswith(('.webp', '.jpg', '.jpeg', '.png')):
            name = Path(image_file).stem
            desc_file = f"descriptions/{name}.txt"
            
            if not os.path.exists(desc_file):
                image_path = os.path.join(season2_path, image_file)
                print(f"\nGenerating description for: {image_file}")
                description = generate_fashion_description(image_path, model)
                
                # Save description to file
                with open(desc_file, 'w', encoding='utf-8') as f:
                    f.write(description)
                print(f"Description saved to {desc_file}")
            else:
                print(f"Description already exists for {image_file}")

if __name__ == "__main__":
    main() 