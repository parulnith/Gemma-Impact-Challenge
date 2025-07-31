"""
Image Processing Module
Handles all image preprocessing operations for form field extraction.
"""

import os
from PIL import Image
from typing import Tuple, Optional
import hashlib


class ImageProcessor:
    """Handles image preprocessing for AI model input"""
    
    def __init__(self, max_size: int = 512):
        """
        Initialize image processor
        
        Args:
            max_size (int): Maximum dimension for image resizing
        """
        self.max_size = max_size
    
    def load_and_validate(self, image_path: str) -> Optional[Image.Image]:
        """
        Load and validate an image file
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            PIL.Image or None: Loaded image or None if failed
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            print(f"[ImageProcessor] Failed to load image: {e}")
            return None
    
    def resize_for_model(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        """
        Resize image for optimal model processing
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            tuple: (resized_image, was_resized)
        """
        original_size = image.size
        
        if image.width > self.max_size or image.height > self.max_size:
            # Create a copy before resizing
            resized_image = image.copy()
            resized_image.thumbnail((self.max_size, self.max_size), Image.Resampling.BICUBIC)
            return resized_image, True
        
        return image, False
    
    def get_image_hash(self, image_path: str) -> Optional[str]:
        """
        Generate MD5 hash for image file
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            str or None: MD5 hash or None if failed
        """
        if not image_path or not os.path.exists(image_path):
            return None
            
        try:
            with open(image_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"[ImageProcessor] Failed to generate hash: {e}")
            return None
    
    def preprocess(self, image_path: str) -> Tuple[Optional[Image.Image], Optional[str], dict]:
        """
        Complete preprocessing pipeline for an image
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            tuple: (processed_image, image_hash, metadata)
        """
        # Generate hash for caching
        image_hash = self.get_image_hash(image_path)
        
        # Load and validate image
        image = self.load_and_validate(image_path)
        if image is None:
            return None, image_hash, {"error": "Failed to load image"}
        
        original_size = image.size
        
        # Resize if needed
        processed_image, was_resized = self.resize_for_model(image)
        
        # Prepare metadata
        metadata = {
            "original_size": original_size,
            "processed_size": processed_image.size,
            "was_resized": was_resized,
            "max_size_limit": self.max_size
        }
        
        return processed_image, image_hash, metadata