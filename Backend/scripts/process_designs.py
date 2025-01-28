import asyncio
from pathlib import Path
import json
from datetime import datetime

import aiohttp
import aiofiles

# Configuration
DESIGNS_DIR = Path("designs")
API_URL = "http://localhost:8000/api/v1/analyze"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

async def analyze_image(session: aiohttp.ClientSession, image_path: Path):
    """Analyze a single image using the API."""
    try:
        async with aiofiles.open(image_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('image',
                          await f.read(),
                          filename=image_path.name,
                          content_type='image/jpeg')
            
            async with session.post(API_URL, data=data) as response:
                return await response.json()
    except Exception as e:
        return {"error": str(e)}

async def process_all_designs():
    """Process all design images in parallel."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for image_path in DESIGNS_DIR.glob("*"):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                task = analyze_image(session, image_path)
                tasks.append((image_path.name, task))
        
        results = {}
        for filename, task in tasks:
            result = await task
            results[filename] = result
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"analysis_results_{timestamp}.json"
        
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(results, indent=2))
        
        print(f"Results saved to {output_file}")
        return results

if __name__ == "__main__":
    asyncio.run(process_all_designs()) 