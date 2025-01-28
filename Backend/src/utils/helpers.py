import os
from pathlib import Path
from typing import List, Union
from PIL import Image
import torch

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists and return its Path object"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def validate_image(image_path: Union[str, Path]) -> bool:
    """Validate if a file is a valid image"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_supported_image_extensions() -> List[str]:
    """Return list of supported image extensions"""
    return ['.jpg', '.jpeg', '.png', '.webp']

def get_device() -> torch.device:
    """Get the appropriate device (CPU/GPU) for torch operations"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_upload_file(upload_file: bytes, destination: Path) -> Path:
    """Save an uploaded file to the specified destination"""
    try:
        with open(destination, 'wb') as f:
            f.write(upload_file)
        return destination
    except Exception as e:
        raise RuntimeError(f"Failed to save uploaded file: {str(e)}")

def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    """Clean up files older than specified hours in the given directory"""
    current_time = Path.ctime(directory)
    for file_path in directory.glob('*'):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_ctime
            if file_age.total_seconds() > max_age_hours * 3600:
                file_path.unlink() 