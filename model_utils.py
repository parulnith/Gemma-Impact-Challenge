"""
Model Loading and Management Utilities for ASHA Form Application

This module handles the loading, caching, and management of the Google Gemma 3n 4B
vision-language model used for form field extraction and audio transcription.

Key Features:
- Smart model loading with automatic download/cache detection
- Memory-efficient model management on CPU
- Fallback handling for offline/online scenarios
- Global model state management with accessor functions

Performance:
- First time: Downloads model (5-10 minutes) + loads (30-60 seconds)
- Subsequent runs: Loads from cache (30-60 seconds)
- During session: Reuses loaded model (instant)

Dependencies:
- transformers: Hugging Face model loading
- torch: PyTorch for model inference
- config: Application configuration
"""

import os

# Model loading and network utilities for Gemma model
import sys
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from config import gemma_model_id, hf_token


# Global model variables - shared across the application
processor = None  # Hugging Face processor for text/image tokenization
model = None      # Gemma 3n 4B vision-language model
device = torch.device("cpu")  # Using CPU for on-device inference

def load_model():
    """
    Load the Gemma 3n 4B model with smart caching and fallback handling.
    
    This function implements intelligent model loading that:
    1. Checks if model is already loaded (avoids reloading)
    2. Attempts download from Hugging Face if not cached
    3. Falls back to offline cached version if download fails
    4. Handles both online and offline scenarios gracefully
    
    Returns:
        bool: True if model loaded successfully, False otherwise
        
    Performance:
        - Already loaded: Instant (skips loading)
        - First time: 5-10 minutes (download) + 30-60s (load)
        - Cached: 30-60 seconds (load from cache)
        
    Raises:
        Prints error messages to stderr on failure
    """
    global processor, model, device
    
    # Check if already loaded - avoid unnecessary reloading
    if model is not None and processor is not None:
        print("Model already loaded, skipping reload.")
        return True
        
    print("Loading model on CPU...")
    
    # Determine cache status and loading strategy
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache_path = os.path.join(cache_dir, f"models--{gemma_model_id.replace('/', '--')}")
    is_cached = os.path.exists(model_cache_path)
    
    # Try online first if model is not cached, then fallback to offline
    use_offline = is_cached
    
    if not is_cached:
        print("Model not found in cache. Attempting to download from Hugging Face...")
        # Clear offline env var for download
        if "TRANSFORMERS_OFFLINE" in os.environ:
            del os.environ["TRANSFORMERS_OFFLINE"]
    else:
        print("Using cached model from local storage.")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        # Load processor and model with appropriate settings
        processor = AutoProcessor.from_pretrained(
            gemma_model_id, 
            token=hf_token, 
            local_files_only=use_offline
        )
        model = AutoModelForImageTextToText.from_pretrained(
            gemma_model_id,
            torch_dtype=torch.float32,      # Use float32 for CPU compatibility
            low_cpu_mem_usage=True,         # Optimize memory usage
            token=hf_token,
            local_files_only=use_offline
        ).to(device).eval()                 # Move to CPU and set to evaluation mode
        
        print(f"Gemma model loaded successfully.")
        return True
        
    except Exception as e:
        if not use_offline:
            print(f"Download failed: {e}")
            print("Retrying with offline mode (cached files only)...")
            # Fallback to offline mode
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                processor = AutoProcessor.from_pretrained(
                    gemma_model_id, 
                    token=hf_token, 
                    local_files_only=True
                )
                model = AutoModelForImageTextToText.from_pretrained(
                    gemma_model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    token=hf_token,
                    local_files_only=True
                ).to(device).eval()
                print(f"Gemma model loaded successfully from cache.")
                return True
            except Exception as e2:
                print(f"CRITICAL: Both online and offline loading failed: {e2}", file=sys.stderr)
                return False
        else:
            print(f"CRITICAL: Model loading failed: {e}", file=sys.stderr)
            return False

def get_model():
    """
    Get the current model instance.
    
    Returns:
        torch.nn.Module or None: The loaded Gemma model or None if not loaded
    """
    return model

def get_processor():
    """
    Get the current processor instance.
    
    Returns:
        AutoProcessor or None: The loaded Hugging Face processor or None if not loaded
    """
    return processor

def get_device():
    """
    Get the current device being used for inference.
    
    Returns:
        torch.device: The device (CPU) used for model inference
    """
    return device

def is_model_loaded():
    """
    Check if both model and processor are loaded and ready for inference.
    
    Returns:
        bool: True if both model and processor are loaded, False otherwise
        
    Usage:
        if is_model_loaded():
            # Safe to use model for inference
            result = process_image(image)
        else:
            # Need to load model first
            load_model()
    """
    return model is not None and processor is not None
