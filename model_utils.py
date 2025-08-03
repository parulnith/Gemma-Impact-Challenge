import os

# Model loading and network utilities for Gemma model
import sys
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from config import gemma_model_id, hf_token


# Global model variables
processor = None
model = None
device = torch.device("cpu")

def load_model():
    global processor, model, device
    
    # Check if already loaded
    if model is not None and processor is not None:
        print("Model already loaded, skipping reload.")
        return True
        
    print("Loading model on CPU...")
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
        processor = AutoProcessor.from_pretrained(gemma_model_id, token=hf_token, local_files_only=use_offline)
        model = AutoModelForImageTextToText.from_pretrained(
            gemma_model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=hf_token,
            local_files_only=use_offline
        ).to(device).eval()
        print(f"Gemma model loaded successfully.")
        return True
    except Exception as e:
        if not use_offline:
            print(f"Download failed: {e}")
            print("Retrying with offline mode (cached files only)...")
            # Fallback to offline mode
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                processor = AutoProcessor.from_pretrained(gemma_model_id, token=hf_token, local_files_only=True)
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
    """Get the current model instance"""
    return model

def get_processor():
    """Get the current processor instance"""
    return processor

def get_device():
    """Get the current device"""
    return device

def is_model_loaded():
    """Check if both model and processor are loaded"""
    return model is not None and processor is not None
