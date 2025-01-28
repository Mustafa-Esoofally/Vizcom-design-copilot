from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from typing import List, Dict

from src.ml.fashion_clip_model import FashionClipModel
from src.utils.helpers import save_upload_file, validate_image, get_supported_image_extensions
from src.config.settings import CACHE_DIR

router = APIRouter()
model = FashionClipModel()

SUPPORTED_CATEGORIES = [
    "t-shirt", "sweatshirt", "hoodie", "jacket",
    "sportswear", "athletic wear", "casual wear",
    "black", "white", "navy", "grey",
    "Under Armour logo", "brand logo",
    "cotton", "fleece", "synthetic fabric"
]

@router.post("/analyze")
async def analyze_image(image: UploadFile = File(...)) -> Dict[str, float]:
    """Analyze an image and return category predictions."""
    try:
        # Validate file extension
        file_ext = Path(image.filename).suffix.lower()
        if file_ext not in get_supported_image_extensions():
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Save uploaded file
        temp_path = CACHE_DIR / f"temp_{image.filename}"
        await save_upload_file(await image.read(), temp_path)
        
        # Validate image
        if not validate_image(temp_path):
            temp_path.unlink()  # Clean up
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Analyze image
        results = model.classify_image(temp_path, SUPPORTED_CATEGORIES)
        
        # Clean up
        temp_path.unlink()
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-analyze")
async def batch_analyze(images: List[UploadFile] = File(...)) -> List[Dict[str, Dict[str, float]]]:
    """Analyze multiple images in batch."""
    results = []
    for image in images:
        try:
            result = await analyze_image(image)
            results.append({image.filename: result})
        except HTTPException as e:
            results.append({image.filename: {"error": str(e.detail)}})
    return results 