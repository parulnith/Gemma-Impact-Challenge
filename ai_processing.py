"""
AI Processing Module for ASHA Form Application

This module contains the core AI processing functions for the ASHA form digitization
application, including form field extraction from images and Hindi audio transcription.

Key Functions:
- extract_fields_from_image(): Extract form field labels from uploaded images
- transcribe_audio(): Convert Hindi speech to Devanagari text
- Caching utilities for performance optimization

AI Models Used:
- Google Gemma 3n 4B: Vision-language model for image understanding
- OpenAI Whisper: For audio transcription in Hindi

Performance Features:
- Hash-based caching for repeated processing
- Smart model loading to avoid reloads
- Progress tracking for long-running operations

Dependencies:
- torch: PyTorch for model inference
- PIL: Image processing
- whisper: OpenAI Whisper for audio transcription
- transformers: Hugging Face model integration
"""

import hashlib
import json
import os
import pickle
import re
import time
from PIL import Image
import torch

# Audio transcription support
try:
    import whisper
except ImportError:
    whisper = None  # Handle gracefully if not installed

# Import functions from our modular components
from model_utils import load_model, get_model, get_processor, get_device
from config import get_model_config

import re
import os
import time
import json
import numpy as np
import soundfile as sf
import torch
import pickle
from PIL import Image
from config import image_cache, audio_cache, cache_file, audio_cache_file, save_to_csv
from model_utils import load_model, get_model, get_processor, get_device, is_model_loaded
import gradio as gr

# Utility functions for caching and data processing
def get_image_hash(image_path):
    """
    Generate MD5 hash for an image file to enable caching.
    
    Creates a unique identifier for image files based on their content,
    allowing the system to cache processing results and avoid reprocessing
    the same image multiple times.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str or None: MD5 hash of the image file, or None if file cannot be read
        
    Example:
        hash_val = get_image_hash("form.jpg")
        # Returns: "a1b2c3d4e5f6..."
    """
    try:
        with open(image_path, "rb") as f:
            content = f.read()
            return hashlib.md5(content).hexdigest()
    except Exception:
        return None


def get_audio_hash(audio_path):
    """
    Generate MD5 hash for an audio file to enable caching.
    
    Creates a unique identifier for audio files based on their content,
    allowing the system to cache transcription results and avoid reprocessing
    the same audio multiple times.
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        str or None: MD5 hash of the audio file, or None if file cannot be read
        
    Example:
        hash_val = get_audio_hash("recording.wav")
        # Returns: "x1y2z3a4b5c6..."
    """
    try:
        with open(audio_path, "rb") as f:
            content = f.read()
            return hashlib.md5(content).hexdigest()
    except Exception:
        return None

def extract_fields_from_image(image_path, progress=None):
    """
    Extract form field labels from an uploaded image using Gemma 3n 4B model.
    
    This is the core AI function that processes photos of handwritten forms
    and extracts field labels (like "Name", "Age", "Village") for digital form creation.
    
    Process:
    1. Load and validate the uploaded image
    2. Check cache for previous processing of same image
    3. Resize image for optimal processing if needed
    4. Use Gemma 3n 4B vision-language model for field extraction
    5. Parse AI output to extract clean field names
    6. Cache results for future use
    
    Args:
        image_path (str): Path to the uploaded form image
        progress (callable, optional): Gradio progress callback for UI updates
        
    Returns:
        list: List of extracted field names (e.g., ["Name", "Age", "Village"])
        
    Performance:
        - Cached results: Instant
        - New images: 30-60 seconds on CPU
        
    Raises:
        Exception: If model loading fails or image processing errors occur
        
    Example:
        fields = extract_fields_from_image("form.jpg")
        # Returns: ["नाम", "आयु", "गाँव", "मोबाइल नंबर"]
    """
    start_time = time.time()
    
    # Ensure model is loaded (will skip if already loaded)
    if not load_model():
        raise Exception("Failed to load model. Please refresh the page and try again.")

    # Get model components
    model = get_model()
    processor = get_processor()
    device = get_device()

    if progress:
        progress(0.05, desc="Starting image processing...")
        
    # Check cache for previous processing of this image
    image_hash = get_image_hash(image_path)
    if image_hash in image_cache:
        cached_fields = image_cache[image_hash]
        if progress:
            for p in [0.3, 0.6, 0.9, 1.0]:
                progress(p, desc="Using cached results...")
                time.sleep(0.05)
        return cached_fields
        
    # Load and validate the image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")
        
    # Optimize image size for processing (balance quality vs speed)
    if image.width > 1024 or image.height > 1024:
        image.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
        
    # Carefully crafted prompt for field extraction
    prompt = (
        "<image_soft_token> Extract ALL possible form field labels from this image. "
        "Return ONLY a JSON object where each key is a field name and each value is 'text'. "
        "For Hindi/Devanagari forms, preserve the original script. "
        "Do NOT include any introductory or concluding text. "
        "If a field is partially visible, still include it. "
        "Sample output: {\"ग्राम\": \"text\", \"उपकेन्द्र\": \"text\", \"आयु\": \"text\", \"लिंग\": \"text\", ...}"
    )
    
    # Process image and text through the model
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    # Generate field extraction using Gemma 3n 4B
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,        # Limit output length for efficiency
            do_sample=False,           # Deterministic output
            num_beams=1,              # Single beam for speed
            temperature=0.1,          # Low temperature for consistency
            repetition_penalty=1.1,   # Prevent repetitive output
            length_penalty=1.0,
            early_stopping=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode model output to text
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Parse AI output to extract field names
    fields = []
    json_match = re.search(r"{.*}", text, re.DOTALL)
    if json_match:
        try:
            parsed_json = json.loads(json_match.group(0))
            fields = list(parsed_json.keys())
        except Exception:
            pass
    
    # Fallback parsing if JSON extraction fails
    if not fields:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            if ":" in line:
                field_name = line.split(":", 1)[0].strip()
                if 2 < len(field_name) < 50:  # Reasonable field name length
                    fields.append(field_name)
    
    # Clean and cache results
    clean_fields = [f.strip() for f in fields if f.strip()]
    if image_hash and clean_fields:
        image_cache[image_hash] = clean_fields
        with open(cache_file, "wb") as f:
            pickle.dump(image_cache, f)
    
    if progress:
        progress(1.0, desc="Extraction complete")
        
    return clean_fields

# Generate a hash for audio data (used for caching)

def transcribe_audio(audio_path, fields, progress=None):
    """
    Convert audio responses to text using OpenAI Whisper model.
    
    This function handles Hindi/English audio transcription for form responses,
    with intelligent caching to improve performance for repeated uploads.
    
    Process:
    1. Validate audio file format and accessibility
    2. Check cache for previous transcription of same audio
    3. Load and configure Whisper model for Hindi/English
    4. Transcribe audio with language detection
    5. Map transcription to corresponding form fields
    6. Cache results for future use
    
    Args:
        audio_path (str): Path to the uploaded audio file (WAV/MP3/M4A)
        fields (list): List of form field names to map responses to
        progress (callable, optional): Gradio progress callback for UI updates
        
    Returns:
        dict: Mapping of field names to transcribed responses
              Example: {"नाम": "राम कुमार", "आयु": "25", "गाँव": "सरोजनी नगर"}
              
    Performance:
        - Cached results: Instant
        - New audio: 10-30 seconds depending on length
        
    Raises:
        Exception: If audio file is invalid, Whisper model fails, or transcription errors occur
        
    Technical Notes:
        - Uses Whisper "base" model for balance of speed and accuracy
        - Automatically detects Hindi/English language
        - Handles various audio formats through librosa
        - Maps responses to fields using simple enumeration
        
    Example:
        fields = ["नाम", "आयु", "गाँव"]
        responses = transcribe_audio("recording.wav", fields)
        # Returns: {"नाम": "राम कुमार", "आयु": "25", "गाँव": "सरोजनी नगर"}
    """
    # Check if Whisper is available
    if whisper is None:
        raise Exception("Whisper not installed. Please install with: pip install openai-whisper")
    
    if progress:
        progress(0.1, desc="Loading audio file...")
    
    # Validate audio file exists and is accessible
    if not os.path.exists(audio_path):
        raise Exception("Audio file not found")
    
    # Check cache for previous transcription of this audio
    audio_hash = get_audio_hash(audio_path)
    if audio_hash in audio_cache:
        cached_result = audio_cache[audio_hash]
        if progress:
            for p in [0.3, 0.6, 0.9, 1.0]:
                progress(p, desc="Using cached transcription...")
                time.sleep(0.05)
        return cached_result
    
    if progress:
        progress(0.3, desc="Loading Whisper model...")
    
    # Load Whisper model for transcription
    try:
        model = whisper.load_model("base")  # Balance of speed and accuracy
    except Exception as e:
        raise Exception(f"Failed to load Whisper model: {str(e)}")
    
    if progress:
        progress(0.5, desc="Transcribing audio...")
    
    # Transcribe audio with automatic language detection
    try:
        result = model.transcribe(
            audio_path,
            language=None,  # Auto-detect Hindi/English
            fp16=False,     # Disable FP16 for compatibility
            verbose=False   # Suppress detailed output
        )
        transcription = result["text"].strip()
    except Exception as e:
        raise Exception(f"Failed to transcribe audio: {str(e)}")
    
    if progress:
        progress(0.8, desc="Processing responses...")
    
    # Split transcription into individual responses
    # Simple approach: split by periods or natural pauses
    responses = [resp.strip() for resp in transcription.split('.') if resp.strip()]
    
    # Map responses to form fields in order
    field_responses = {}
    for i, field in enumerate(fields):
        if i < len(responses):
            field_responses[field] = responses[i]
        else:
            field_responses[field] = ""  # Empty if no corresponding response
    
    # Cache the result for future use
    if audio_hash:
        audio_cache[audio_hash] = field_responses
        with open(cache_file, "wb") as f:
            pickle.dump({"image_cache": image_cache, "audio_cache": audio_cache}, f)
    
    if progress:
        progress(1.0, desc="Transcription complete")
    
    return field_responses
