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

# Utility functions
def get_image_hash(image_path):
    if not image_path or not os.path.exists(image_path): return None
    with open(image_path, "rb") as f: return hash(f.read())

def get_audio_hash(audio_data):
    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sr, audio_np_array = audio_data
        return hash(audio_np_array.tobytes() + str(sr).encode())
    return None

def extract_fields_from_image(image_path, progress=None):
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
    image_hash = get_image_hash(image_path)
    if image_hash in image_cache:
        cached_fields = image_cache[image_hash]
        if progress:
            for p in [0.3, 0.6, 0.9, 1.0]:
                progress(p, desc="Using cached results...")
                time.sleep(0.05)
        return cached_fields
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")
    if image.width > 1024 or image.height > 1024:
        image.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
    prompt = ("<image_soft_token> Extract ALL possible form field labels from this image. "
              "Return ONLY a JSON object where each key is a field name and each value is 'text'. "
              "For Hindi/Devanagari forms, preserve the original script. "
              "Do NOT include any introductory or concluding text. "
              "If a field is partially visible, still include it. "
              "Sample output: {\"ग्राम\": \"text\", \"उपकेन्द्र\": \"text\", \"आयु\": \"text\", \"लिंग\": \"text\", ...}")
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            temperature=0.1,
            repetition_penalty=1.1,
            length_penalty=1.0,
            early_stopping=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    fields = []
    json_match = re.search(r"{.*}", text, re.DOTALL)
    if json_match:
        try:
            parsed_json = json.loads(json_match.group(0))
            fields = list(parsed_json.keys())
        except Exception:
            pass
    if not fields:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            if ":" in line:
                field_name = line.split(":", 1)[0].strip()
                if 2 < len(field_name) < 50:
                    fields.append(field_name)
    clean_fields = [f.strip() for f in fields if f.strip()]
    if image_hash and clean_fields:
        image_cache[image_hash] = clean_fields
        with open(cache_file, "wb") as f:
            pickle.dump(image_cache, f)
    if progress:
        progress(1.0, desc="Extraction complete")
    return clean_fields

# Generate a hash for audio data (used for caching)

def transcribe_audio(audio_data, progress=gr.Progress()):
    if progress:
        progress(0.05, desc="Starting audio transcription...")
    audio_hash = get_audio_hash(audio_data)
    if audio_hash and audio_hash in audio_cache:
        cached_result = audio_cache[audio_hash]
        if progress:
            progress(1.0, desc="Transcription complete!")
        return cached_result
    
    # Ensure model is loaded (will skip if already loaded)
    if not load_model():
        return "[Error] Failed to load Gemma model."
    
    # Get model components
    model = get_model()
    processor = get_processor()
    device = get_device()
    
    if audio_data is None:
        return "[Error] No audio provided."
    audio_np_array, sr = (None, None)
    if isinstance(audio_data, str):
        audio_np_array, sr = sf.read(audio_data)
    elif isinstance(audio_data, tuple) and len(audio_data) == 2:
        sr, audio_np_array = audio_data
    else:
        return "[Error] Invalid audio input format."
    if audio_np_array is None or len(audio_np_array) == 0:
        return "[Error] Empty audio data."
    if audio_np_array.ndim > 1:
        audio_np_array = np.mean(audio_np_array, axis=1)
    if np.max(np.abs(audio_np_array)) > 0:
        audio_np_array = audio_np_array / (np.max(np.abs(audio_np_array)) + 1e-9)
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
    input_dict = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        sampling_rate=sr
    )
    final_inputs_for_model = {k: v.to(model.device) for k, v in input_dict.items()}
    with torch.inference_mode():
        predicted_ids = model.generate(
            **final_inputs_for_model,
            max_new_tokens=32,
            do_sample=False,
            num_beams=1
        )
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0].strip()
    if not transcription or transcription.strip() == "":
        transcription = "Audio not clear."
    if audio_hash and transcription:
        audio_cache[audio_hash] = transcription
        with open(audio_cache_file, "wb") as f:
            pickle.dump(audio_cache, f)
    if progress:
        progress(1.0, desc="Transcription complete!")
    return transcription
