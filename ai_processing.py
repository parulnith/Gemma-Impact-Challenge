"""
AI Processing Module for ASHA Form Application

This module contains the core AI processing functions for the ASHA form digitization
application, including form field extraction from images and Hindi audio transcription.

Key Functions:
- extract_fields_from_image(): Extract form field labels from uploaded images
- transcribe_audio(): Convert Hindi speech to Devanagari text using Gemma
- Image caching utilities for performance optimization

AI Models Used:
- Google Gemma 3n 4B: Vision-language model for image understanding and audio transcription

Performance Features:
- Image hash-based caching for repeated form template processing
- Smart model loading to avoid reloads
- Progress tracking for long-running operations
"""

import hashlib
import json
import os
import pickle
import re
import time
import numpy as np
import soundfile as sf
from PIL import Image
import torch

# Import functions from our modular components
from model_utils import load_model, get_model, get_processor, get_device, is_model_loaded
from config import image_cache, cache_file, save_to_csv

# Define global cache for performance optimization (images only)
cache_file = "cache.pkl"
image_cache = {}

# Load existing image cache on startup
if os.path.exists(cache_file):
    try:
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)
            if isinstance(cache_data, dict):
                image_cache = cache_data.get("image_cache", {})
    except Exception:
        # If cache loading fails, start with empty cache
        image_cache = {}

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
        "<image_soft_token> Carefully examine this form image and extract ALL visible form field labels, including partially visible ones. "
        "Return ONLY a complete JSON object where each key is a field name and each value is 'text'. "
        "For Hindi/Devanagari forms, preserve the original script exactly. "
        "Include every single field label you can see, even if it's cut off or partially visible. "
        "Do NOT include any introductory text, explanations, or conclusions. "
        "Extract ALL fields - aim for 15-25 fields if they exist in the image. "
        "Sample format: {\"ग्राम\": \"text\", \"उपकेन्द्र\": \"text\", \"आयु\": \"text\", \"लिंग\": \"text\", \"नाम\": \"text\", \"पता\": \"text\", \"मोबाइल\": \"text\", ...}"
    )
    
    # Process image and text through the model
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    # Generate field extraction using Gemma 3n 4B
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,        # Significantly increased for comprehensive extraction (20-30+ fields)
            do_sample=False,           # Deterministic output
            num_beams=1,              # Single beam for speed
            repetition_penalty=1.1,   # Prevent repetitive output
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
    
    # Enhanced fallback parsing if JSON extraction fails or incomplete
    if not fields or len(fields) < 10:  # If we got too few fields, try alternative parsing
        # Try to find quoted field names
        quoted_fields = re.findall(r'"([^"]+)":\s*"text"', text, re.IGNORECASE)
        if quoted_fields:
            fields.extend(quoted_fields)
        
        # Try to find fields in colon format
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            if ":" in line:
                field_name = line.split(":", 1)[0].strip().strip('"')
                if 2 < len(field_name) < 50:  # Reasonable field name length
                    fields.append(field_name)
        
        # Try to find fields separated by commas
        comma_fields = re.findall(r'"([^"]+)"', text)
        for field in comma_fields:
            if 2 < len(field) < 50 and field not in ['text', 'label', 'field']:
                fields.append(field)
    
    # Clean and cache results - remove duplicates while preserving order
    seen = set()
    clean_fields = []
    for field in fields:
        field_clean = field.strip()
        if field_clean and field_clean not in seen and len(field_clean) > 1:
            clean_fields.append(field_clean)
            seen.add(field_clean)
    
    # Debug info (can be removed in production)
    print(f"[DEBUG] Extracted {len(clean_fields)} fields: {clean_fields[:5]}..." if clean_fields else "[DEBUG] No fields extracted")
    
    if image_hash and clean_fields:
        image_cache[image_hash] = clean_fields
        with open(cache_file, "wb") as f:
            pickle.dump(image_cache, f)
    
    if progress:
        progress(1.0, desc="Extraction complete")
        
    return clean_fields

def transcribe_audio(audio_data, fields=None, progress=None):
    """
    Fast audio transcription using Gemma 3n 4B for STT without caching.
    
    This function handles Hindi/English audio transcription using the same Gemma model
    that's used for image processing, providing consistent AI capabilities across the app.
    Audio responses are not cached since each form will have unique responses.
    
    Process:
    1. Validate and process audio input (file path or tuple format)
    2. Normalize and resample audio to target format
    3. Use Gemma 3n 4B model with audio processing capabilities
    4. Clean and post-process transcription output
    
    Args:
        audio_data: Either audio file path (str) or tuple (sample_rate, audio_array)
        fields (list, optional): List of form field names (for compatibility, not used in transcription)
        progress (callable, optional): Gradio progress callback for UI updates
        
    Returns:
        str: Transcribed text in Hindi/English
        
    Performance:
        - New audio: 10-30 seconds depending on length
        - No caching to ensure fresh transcription for each form
        
    Raises:
        Exception: If audio processing fails or Gemma model errors occur
        
    Example:
        transcription = transcribe_audio("recording.wav")
        # Returns: "नाम राम कुमार आयु पच्चीस गाँव सरोजनी नगर"
    """
    if progress:
        progress(0.05, desc="Starting audio transcription...")
    
    # Ensure model is loaded (will skip if already loaded)
    if not load_model():
        raise Exception("Failed to load model. Please refresh the page and try again.")

    # Get model components
    model = get_model()
    processor = get_processor()
    device = get_device()
    
    if model is None or processor is None:
        raise Exception("Gemma model or processor not loaded properly.")
    
    if audio_data is None:
        raise Exception("No audio provided.")
    
    # Process audio input - handle both file paths and tuple format
    audio_np_array, sr = (None, None)
    if isinstance(audio_data, str):
        # Audio file path
        try:
            audio_np_array, sr = sf.read(audio_data)
        except Exception as e:
            raise Exception(f"Failed to read audio file: {str(e)}")
    elif isinstance(audio_data, tuple) and len(audio_data) == 2:
        # Gradio audio format: (sample_rate, audio_array)
        sr, audio_np_array = audio_data
    else:
        raise Exception("Invalid audio input format.")
    
    if audio_np_array is None or len(audio_np_array) == 0:
        raise Exception("Empty audio data.")
    
    if progress:
        progress(0.2, desc="Processing audio...")
    
    # Convert stereo to mono if needed
    if audio_np_array.ndim > 1:
        audio_np_array = np.mean(audio_np_array, axis=1)
    
    # Normalize audio
    if np.max(np.abs(audio_np_array)) > 0:
        audio_np_array = audio_np_array / (np.max(np.abs(audio_np_array)) + 1e-9)
    
    # Resample to target sample rate (16kHz for optimal processing)
    target_sr = 16000
    if sr != target_sr:
        ratio = target_sr / sr
        new_length = int(len(audio_np_array) * ratio)
        audio_np_array = np.interp(
            np.linspace(0, len(audio_np_array), new_length),
            np.arange(len(audio_np_array)),
            audio_np_array
        )
        sr = target_sr
    
    # Prepare prompt and messages for Gemma audio processing
    prompt = "Transcribe this audio in Hindi (Devanagari script)."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_np_array},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    if progress:
        progress(0.4, desc="Tokenizing audio and prompt...")
    
    # Process audio and text through Gemma
    try:
        input_dict = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            sampling_rate=sr
        )
        final_inputs_for_model = {k: v.to(model.device) for k, v in input_dict.items()}
    except Exception as e:
        raise Exception(f"Failed to process audio with Gemma: {str(e)}")
    
    if progress:
        progress(0.6, desc="Generating transcription...")
    
    # Generate transcription using Gemma 3n 4B
    with torch.inference_mode():
        predicted_ids = model.generate(
            **final_inputs_for_model,
            max_new_tokens=64,  # Sufficient for typical form responses
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1
        )
    
    if progress:
        progress(0.9, desc="Decoding transcription...")
    
    # Decode the transcription
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0].strip()
    
    # Clean up the transcription output
    # Remove any model/user tokens
    transcription = re.sub(r'\b(user|model)\s*[:：\-]+', '', transcription, flags=re.IGNORECASE)
    transcription = re.sub(r'\b(user|model)\b', '', transcription, flags=re.IGNORECASE)
    
    # Remove the prompt text from output
    prompt_text = "Transcribe this audio in Hindi (Devanagari script)."
    transcription = transcription.replace(prompt_text, '')
    
    # Clean up whitespace and punctuation
    transcription = transcription.strip(' :\n\t-')
    
    # Handle empty or unclear transcription
    if not transcription or transcription.strip() == "":
        transcription = "Audio not clear."
    
    if progress:
        progress(1.0, desc="Transcription complete!")
    
    return transcription
