import sys
import platform
import re
import os
import json
import time
import hashlib
import pickle
import warnings
import gradio as gr
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, TextStreamer
from PIL import Image
import torchaudio
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
import psutil
import gc
import tempfile
import requests
from huggingface_hub import HfApi

# --- Configuration ---
GEMMA_MODEL_ID = "google/gemma-3n-E4B-it"
CACHE_FILE = "image_cache.pkl"
AUDIO_CACHE_FILE = "audio_cache.pkl"

# --- Cache Loading ---
def load_cache(file_path):
    """Loads a cache file if it exists, otherwise returns an empty dictionary."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            # Keep this print as it's an important warning about a failed operation.
            print(f"[Cache] Could not load cache from {file_path}: {e}")
    return {}

image_cache = load_cache(CACHE_FILE)
audio_cache = load_cache(AUDIO_CACHE_FILE)

# --- Hugging Face Token & Network Check ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    warnings.warn("HF_TOKEN not found in .env file. This may cause issues if models aren't cached.")

def check_network_connectivity():
    """Checks for a connection to Hugging Face servers."""
    try:
        requests.get("https://huggingface.co", timeout=5)
        return True
    except requests.ConnectionError:
        return False

# --- Global Model Variables ---
processor = None
model = None
device = torch.device("cpu")

def load_model_with_fallback():
    """Loads the Gemma model, handling offline mode and caching."""
    global processor, model, device
    
    if model and model != "cached": return True
    
    print("Loading model on CPU...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache_path = os.path.join(cache_dir, f"models--{GEMMA_MODEL_ID.replace('/', '--')}")
    
    is_cached = os.path.exists(model_cache_path)
    has_network = check_network_connectivity()
    use_offline = is_cached and not has_network
    
    if use_offline:
        print("Network unavailable. Using offline mode with cached model.")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
    try:
        processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, token=hf_token, local_files_only=use_offline)
        model = AutoModelForImageTextToText.from_pretrained(
            GEMMA_MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=hf_token,
            local_files_only=use_offline
        ).to(device).eval()
        print(f"Gemma model loaded successfully.")
        return True
    except Exception as e:
        print(f"CRITICAL: Model loading failed: {e}", file=sys.stderr)
        return False


# --- Utility Functions ---
class TokenStreamHandler(TextStreamer):
    """A silent streamer that updates a Gradio progress bar."""
    def __init__(self, tokenizer, progress_callback=None):
        super().__init__(tokenizer)
        self.progress_callback = progress_callback
        self.tokens_generated = 0

    def put(self, value):
        super().put(value)
        self.tokens_generated += 1
        if self.progress_callback:
            progress_estimate = min(0.5 + (self.tokens_generated / 200) * 0.4, 0.9)
            self.progress_callback(progress_estimate)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if stream_end and self.progress_callback:
            self.progress_callback(0.95)

def get_image_hash(image_path):
    if not image_path or not os.path.exists(image_path): return None
    with open(image_path, "rb") as f: return hashlib.md5(f.read()).hexdigest()

def get_audio_hash(audio_data):
    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sr, audio_np_array = audio_data
        return hashlib.md5(audio_np_array.tobytes() + str(sr).encode()).hexdigest()
    return None

def save_to_csv(data, csv_path="output/forms.csv"):
    """Saves form data to a CSV file in long format."""
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        rows_to_save = [
            {"Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "Field_Name": field, "Field_Value": value}
            for field, value in data.items()
        ]
        
        df = pd.DataFrame(rows_to_save)
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", header=header, index=False)
        # Keep this print as it confirms a critical user action succeeded.
        print(f"[CSV] Data for {len(rows_to_save)} fields saved to {csv_path}")
    except Exception as e:
        raise gr.Error(f"Failed to save to CSV: {e}")

# =============================================================================
# CORE AI PROCESSING FUNCTIONS - Main logic for form field extraction
# =============================================================================

def extract_fields_from_image(image_path, progress=None):
    """
    Main extraction function - Extracts form field labels from images using Gemma 3n 4B
    
    This is the core AI function that:
    1. Loads and preprocesses form images (photos of paper forms)
    2. Uses Google's Gemma 3n vision-language model to identify field labels
    3. Parses the AI output to extract clean field names
    4. Caches results to avoid re-processing the same image
    
    Args:
        image_path (str): Path to uploaded form image
        progress (callable): Gradio progress callback for UI updates
        
    Returns:
        list: Clean list of extracted field names (e.g., ["Name", "Age", "Village"])
        
    Performance: 30-60 seconds on CPU for 4B model
    """
    global model, processor, device
    start_time = time.time()
    
    # Console logging for monitoring AI processing pipeline
    print(f"\n{'='*60}")
    print(f"[EXTRACTION START] Processing: {os.path.basename(image_path) if image_path else 'None'}")
    print(f"[EXTRACTION START] Timestamp: {time.strftime('%H:%M:%S')}")
    print(f"[EXTRACTION START] Using Gemma 3n 4B Vision-Language Model")
    print(f"{'='*60}")
    
    # Lazy loading: Load model only when first needed
    if model == "cached":
        print("[Model] Loading Gemma 3n 4B model on first use...")
        if not load_model_with_fallback():
            return ["[Error] Failed to load model on first use."]

    if progress:
        progress(0.05, desc="Starting image processing...")
    print("[Status] AI extraction pipeline initialized")

    # Smart caching: Check if we've processed this exact image before
    print("[Cache] Checking for previously processed results...")
    image_hash = get_image_hash(image_path)
    print(f"[Cache] Image hash: {image_hash[:12] if image_hash else 'None'}...")
    
    if image_hash in image_cache:
        cached_fields = image_cache[image_hash]
        elapsed = time.time() - start_time
        print(f"[Cache] CACHE HIT! Found {len(cached_fields)} pre-extracted fields in {elapsed:.2f}s")
        print(f"[Cache] Cached fields: {cached_fields}")
        if progress:
            for p in [0.3, 0.6, 0.9, 1.0]:
                progress(p, desc="Using cached results...")
                time.sleep(0.05)
        return cached_fields

    print("[Cache] New image detected - proceeding with fresh extraction...")

    # Image preprocessing: Load and validate the uploaded form image
    print("[Image] Loading and validating uploaded form...")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"[Image] Successfully loaded! Original dimensions: {image.size}")
    except Exception as e:
        print(f"[Image] Failed to load image: {e}")
        return []

    if progress:
        progress(0.15, desc="Optimizing image for processing...")

    # Smart resizing: Balance quality vs speed for CPU processing
    original_size = image.size
    max_size = 1024
    if image.width > max_size or image.height > max_size:
        print(f"[Image] Resizing from {image.size} to fit {max_size}px...")
        image.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
        print(f"[Image] Optimized to: {image.size}")
    else:
        print(f"[Image] Size acceptable, keeping: {image.size}")

    if progress:
        progress(0.25, desc="Preparing prompt...")

    # Prompt engineering: Carefully crafted prompt for multilingual form extraction
    prompt = (
    "<image_soft_token> Extract ALL possible form field labels from this image. "
    "Return ONLY a JSON object where each key is a field name and each value is 'text'. "
    "For Hindi/Devanagari forms, preserve the original script. "
    "Do NOT include any introductory or concluding text. "
    "If a field is partially visible, still include it. "
    "Sample output: {\"ग्राम\": \"text\", \"उपकेन्द्र\": \"text\", \"आयु\": \"text\", \"लिंग\": \"text\", ...}"
    )
    #prompt = "<image_soft_token> Extract all the form field labels from this image. Return ONLY a JSON object where keys are field names and values are 'text'. For Hindi/Devanagari forms, preserve the original script. Do NOT include any introductory or concluding text."
    print(f"[Prompt] Using engineered prompt: {prompt[:80]}...")


    if progress:
        progress(0.35, desc="Tokenizing inputs...")

    # Tokenization: Convert image and text to model input format
    print("[Tokenizer] Converting image and text to neural network format...")
    tokenize_start = time.time()
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    tokenize_time = time.time() - tokenize_start
    print(f"[Tokenizer] Tokenization completed in {tokenize_time:.2f}s")
    print(f"[Tokenizer] Input shapes: {', '.join([f'{k}: {v.shape}' for k, v in inputs.items() if hasattr(v, 'shape')])}")

    if progress:
        progress(0.50, desc="Running AI model (CPU processing)...")

    # Model inference: Run Gemma 3n 4B on the form image
    print("[Model] Starting Gemma 3n 4B inference...")
    print("[Model] CPU Processing: Expected time 30-60 seconds")
    generation_start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,      # Optimized for CPU speed
            do_sample=False,         # Deterministic output
            num_beams=1,            # Single beam for efficiency
            temperature=0.1,         # Low temperature for consistency
            repetition_penalty=1.1,  # Prevent repetition
            length_penalty=1.0,
            early_stopping=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    generation_time = time.time() - generation_start
    print(f"[Model] Inference complete! Processing time: {generation_time:.2f}s")

    # Output decoding: Convert model output back to readable text
    print("[Decoder] Converting model output to text...")
    decode_start = time.time()
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    decode_time = time.time() - decode_start
    print(f"[Decoder] Decoding completed in {decode_time:.3f}s")
    print(f"[Output] Raw AI response preview: {text[:200]}...")

    # JSON parsing: Extract structured field names from AI response
    print("[Parser] Analyzing AI response for field data...")
    fields = []
    json_match = re.search(r"{.*}", text, re.DOTALL)
    json_str = None
    
    if json_match:
        json_str = json_match.group(0)
        print(f"[Parser] Found JSON structure: {json_str[:100]}...")
        try:
            parsed_json = json.loads(json_str)
            print(f"[Parser] JSON parsed successfully, {len(parsed_json)} field definitions found")
            
            # Handle nested JSON structures
            def flatten_json(json_obj):
                """Recursively flatten JSON to extract field names"""
                flat = {}
                for k, v in json_obj.items():
                    if isinstance(v, dict) and "type" in v:
                        flat[k] = v["type"]
                    elif isinstance(v, dict):
                        flat.update(flatten_json(v))
                    else:
                        flat[k] = v
                return flat
            
            flat_json = flatten_json(parsed_json)
            fields = list(flat_json.keys())
            print(f"[Parser] Extracted {len(fields)} fields: {fields}")
        except Exception as e:
            print(f"[Parser] JSON parsing failed: {e}")
    else:
        print("[Parser] No JSON structure detected in output")

    # Fallback extraction: Use regex patterns if JSON parsing fails
    if not fields:
        print("[Fallback] Using regex-based field extraction...")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        numbered_pattern = re.compile(r'^\s*(\d+)[\.|\)]?\s*(.+?)(?:\s*[:：](.*))?$')
        fallback_fields = []
        
        for i, line in enumerate(lines):
            # Skip instructional text from AI
            if "extract" in line.lower() or "json" in line.lower() or "example" in line.lower():
                continue
                
            field_name = None
            match = numbered_pattern.match(line)
            if match:
                raw_field = match.group(2).strip()
                field_name = raw_field.rstrip(':：')
            elif ":" in line:
                field_name = line.split(":", 1)[0].strip()
                
            # Quality filter for reasonable field names
            if field_name and 2 < len(field_name) < 50:
                fallback_fields.append(field_name)
                print(f"[Fallback] Line {i+1}: '{line}' -> '{field_name}'")
        
        fields = fallback_fields
        print(f"[Fallback] Extracted {len(fields)} fields using pattern matching")

    # Final processing: Clean up results
    clean_fields = [f.strip() for f in fields if f.strip()]
    total_time = time.time() - start_time
    
    # Results summary
    print(f"\n[RESULT] Successfully extracted {len(clean_fields)} form fields:")
    for i, field in enumerate(clean_fields, 1):
        print(f"[RESULT]   {i}. {field}")
    
    # Cache results for future use
    if image_hash and clean_fields:
        print(f"[Cache] Saving {len(clean_fields)} extracted fields to cache...")
        image_cache[image_hash] = clean_fields
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(image_cache, f)
        print("[Cache] Results cached successfully")

    # Performance summary
    print(f"\n{'='*60}")
    print(f"[EXTRACTION COMPLETE] Total time: {total_time:.2f}s")
    print(f"[EXTRACTION COMPLETE] Status: {'SUCCESS' if len(clean_fields) > 0 else 'NO FIELDS FOUND'}")
    print(f"[EXTRACTION COMPLETE] Ready for digital form filling")
    print(f"{'='*60}\n")

    if progress:
        progress(1.0, desc="Extraction complete")
    return clean_fields

def transcribe_audio(audio_data, progress=gr.Progress()):
    """Fast audio transcription using Gemma for STT with caching"""
    global model, processor, device, audio_cache
    if progress:
        progress(0.05, desc="Starting audio transcription...")
    audio_hash = get_audio_hash(audio_data)
    if audio_hash and audio_hash in audio_cache:
        cached_result = audio_cache[audio_hash]
        if progress:
            progress(1.0, desc="Transcription complete!")
        return cached_result
    if model == "cached":
        if not load_model_with_fallback():
            return "[Error] Failed to load Gemma model on first use."
    if model is None or processor is None:
        return "[Error] Gemma model or processor not loaded properly."
    if audio_data is None:
        return "[Error] No audio provided."
    # manage_memory()  # Uncomment if you have a manage_memory function
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
    if progress:
        progress(0.2, desc="Tokenizing audio and prompt...")
    input_dict = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        sampling_rate=sr
    )
    final_inputs_for_model = {k: v.to(model.device) for k, v in input_dict.items()}
    if progress:
        progress(0.6, desc="Generating transcription...")
    with torch.inference_mode():
        predicted_ids = model.generate(
            **final_inputs_for_model,
            max_new_tokens=32,  # Lowered for faster short-form transcription
            do_sample=False,
            num_beams=1
        )
    if progress:
        progress(0.9, desc="Decoding transcription...")
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0].strip()
    # Remove any 'user:' or 'model:' tokens anywhere in the output
    transcription = re.sub(r'\b(user|model)\s*[:：\-]+', '', transcription, flags=re.IGNORECASE)
    # Remove standalone 'user' or 'model' words (with or without surrounding whitespace)
    transcription = re.sub(r'\b(user|model)\b', '', transcription, flags=re.IGNORECASE)
    # Remove the prompt text anywhere in the output (not just at the start)
    prompt_text = "Transcribe this audio in Hindi (Devanagari script)."
    transcription = transcription.replace(prompt_text, '')
    # Remove extra whitespace and leading/trailing punctuation
    transcription = transcription.strip(' :\n\t-')
    if not transcription or transcription.strip() == "":
        transcription = "Audio not clear."
    if audio_hash and transcription:
        audio_cache[audio_hash] = transcription
        with open(AUDIO_CACHE_FILE, "wb") as f:
            pickle.dump(audio_cache, f)
    if progress:
        progress(1.0, desc="Transcription complete!")
    return transcription

# --- Gradio UI ---
def main_ui():
    """Builds the main Gradio user interface."""

    with gr.Blocks(theme=gr.themes.Soft(), css="""
    /* Apple-like system font stack */
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'San Francisco', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
        background: #fff !important; /* white background */
    }
    .skyblue-btn button {
        background: #87ceeb !important;
        color: #1565c0 !important;
        border: none !important;
        font-weight: 600 !important;
        transition: background 0.2s;
    }
    .skyblue-btn button:disabled {
        background: #b3e5fc !important;
        color: #90a4ae !important;
    }
    .skyblue-btn button:hover:not(:disabled) {
        background: #4fc3f7 !important;
        color: #0d47a1 !important;
    }
    .asha-title {
        font-family: -apple-system, BlinkMacSystemFont, 'San Francisco', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1565c0; /* deeper skyblue for title */
        letter-spacing: 0.5px;
        margin-bottom: 0.5em;
        text-align: center;
    }
    .asha-subtitle {
        font-family: -apple-system, BlinkMacSystemFont, 'San Francisco', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;
        font-size: 1.2rem;
        color: #1976d2;
        margin-bottom: 1.2em;
        text-align: center;
    }
    .asha-section { background: #e3f2fd; border-radius: 14px; box-shadow: 0 2px 12px #b3e5fc; padding: 2.2em 2em 1.5em 2em; margin-bottom: 1.5em; }
    .asha-list { font-size: 1.08rem; color: #263238; margin-bottom: 1.2em; }
    .asha-list li { margin-bottom: 0.5em; }
    .asha-note { color: #01579b; font-size: 1.05rem; font-style: italic; margin-top: 1em; }
    .tab-disabled { opacity: 0.5 !important; pointer-events: none !important; }
    .tab-enabled { opacity: 1 !important; }
    .sample-img { border: 2px solid #b3e5fc; border-radius: 8px; margin: 4px; max-width: 120px; cursor: pointer; transition: border 0.2s; }
    .sample-img:hover { border: 2px solid #0288d1; }
    .cancel-audio-x {
        display: inline-block;
        color: #d32f2f;
        font-size: 1.3em;
        font-weight: bold;
        cursor: pointer;
        margin-left: 0.5em;
        vertical-align: middle;
        user-select: none;
        transition: color 0.2s;
    }
    .cancel-audio-x:hover {
        color: #b71c1c;
    }
    /* Icon-style button for cancel audio */
    .cancel-audio-x-btn button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #d32f2f !important;
        font-size: 1.3em !important;
        font-weight: bold !important;
        padding: 0 0.3em !important;
        margin-left: 0.5em !important;
        vertical-align: middle !important;
        cursor: pointer !important;
        min-width: 1.7em !important;
        min-height: 1.7em !important;
        border-radius: 50% !important;
        transition: color 0.2s !important;
    }
    .cancel-audio-x-btn button:hover {
        color: #b71c1c !important;
        background: #fbe9e7 !important;
    }
    """) as demo:
        fields_state = gr.State([])
        extraction_in_progress = gr.State(False)
        MAX_FIELDS = 15

        # --- Tab Gating State ---
        image_uploaded = gr.State(False)
        fields_extracted = gr.State(False)

        # --- Tab UI ---
        with gr.Tabs() as tabs:
            # --- About Tab ---
            with gr.TabItem("1. About & Demo Info", id=0):
                with gr.Column(elem_classes=["asha-section"]):
                    gr.Markdown(
                        """
# ON DEVICE SMART ASHA Form 
### **Digitizing rural health, empowering ASHA workers with AI.**

This application is designed to help **ASHA (Accredited Social Health Activist) workers**—the backbone of India's rural healthcare system—quickly digitize handwritten forms and transcribe Hindi voice input. ASHA workers are often the first point of contact for healthcare in villages, but their work is slowed by manual paperwork and language barriers. This tool streamlines their workflow, making data entry faster, more accurate, and accessible even for those more comfortable with Hindi speech than typing.

- **Image-based field extraction:** Upload a photo of an ASHA form and the app will automatically detect and extract all field labels, ready for digital entry.  
- **Hindi voice transcription:** Fill any field by speaking in Hindi (Devanagari script) for instant, accurate transcription.  
- **Data export:** All submitted data is saved in a CSV for further use or analysis.  

## Powered by Gemma-3n-4B 
This demo runs on the **Gemma 3n 4B model**, a lightweight yet capable model for on-device AI tasks like image-to-text and Hindi speech transcription.  

## On-device CPU Loading
Since this application runs entirely **on-device using CPU**, the model takes some time to **load the first time**. After the initial load, processing is smooth and does not require an internet connection.

## Why this matters
ASHA workers serve over 900 million people in rural India, often with limited digital literacy and resources. By making form digitization and voice transcription seamless, this app saves time, reduces errors, and helps bring rural health data into the digital age—empowering both workers and the communities they serve.

*Note:*
While you are reading this, the **Gemma 3n model** is being loaded in the background to ensure a smooth and fast demo experience. Please explore each step—the workflow is strictly gated for demo clarity. All features are designed for real-world usability and hackathon evaluation.
                        """,
                        elem_id="asha_about"
                    )
                    model_status = gr.Textbox(value="Loading model in background...", interactive=False, show_label=False, visible=True)

            # --- Upload Tab ---
            with gr.TabItem("2. Upload Image | छवि अपलोड करें", id=1):
                with gr.Row():
                    with gr.Column(scale=2):
                        image_input = gr.Image(type="filepath", label="Upload Form Image | फॉर्म की छवि अपलोड करें", height=350, sources=["upload"])  # Only allow file upload
                        with gr.Row():
                            extract_btn = gr.Button("Extract Fields | फ़ील्ड निकालें", variant="secondary", elem_classes=["skyblue-btn"])
                        gr.Markdown("**Or try a sample image | या नमूना छवि आज़माएँ:**", elem_id="sample-image-label")
                        with gr.Row():
                            sample_gallery = gr.Gallery(
                                value=[
                                   
                                    "samples/sample_form_2.png",
                                   
                                ],
                                label=None,
                                show_label=False,
                                elem_id="sample-gallery",
                                height=110,
                                columns=[3],
                                object_fit="contain",
                                allow_preview=True,
                                interactive=True,
                                elem_classes=["sample-img"]
                            )
                    # Removed status_box column

            # --- Fill Form Tab ---
            with gr.TabItem("3. Fill Form | फॉर्म भरें", id=2):
                form_placeholder = gr.Markdown("""Please extract fields from Step 2 first |कृपया पहले चरण 2 से फ़ील्ड निकालें।""", visible=True)
                with gr.Column(visible=False) as form_container:
                    text_inputs, audio_inputs, field_rows = [], [], []
                    for i in range(MAX_FIELDS):
                        with gr.Row(visible=False) as row:
                            text_input = gr.Textbox(interactive=True, label="Enter value | मान दर्ज करें")
                            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Speak to fill | बोलें और भरें", streaming=False)
                            text_inputs.append(text_input)
                            audio_inputs.append(audio_input)
                            field_rows.append(row)
                            audio_input.change(fn=transcribe_audio, inputs=audio_input, outputs=text_input, show_progress="full")

                with gr.Row(visible=False) as action_row:
                    submit_btn = gr.Button("Submit Form | फॉर्म सबमिट करें", variant="primary")
                    new_form_btn = gr.Button("Start New Form | नया फॉर्म शुरू करें")
        # If you have a record_again_btn, add cancel_previous=True to its click event as well
        # Example:
        # record_again_btn.click(fn=on_record_again, inputs=[], outputs=[text_input, record_again_btn], cancel_previous=True)
        # Removed cancel_x buttons and their logic

        # --- Model Preload on App Start (Tab 1) ---
        def preload_model():
            ok = load_model_with_fallback()
            return gr.update(value="Model loaded and ready!" if ok else "Model failed to load. Please check setup.")

        # Schedule model loading as soon as the app starts (Tab 1 is shown)
        demo.load(fn=preload_model, inputs=None, outputs=model_status, queue=False)

        # --- UI Callback Functions ---
        def clear_cache_and_reset():
            global image_cache, audio_cache
            image_cache.clear()
            audio_cache.clear()
            if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
            if os.path.exists(AUDIO_CACHE_FILE): os.remove(AUDIO_CACHE_FILE)
            return "All caches cleared successfully!"

        def on_sample_click(evt: gr.SelectData):
            # evt.value can be a dict or a string path
            value = evt.value
            if isinstance(value, dict) and "image" in value and "path" in value["image"]:
                value = value["image"]["path"]
            return gr.update(value=value), True

        # Dynamically determine the number of outputs for extract_btn
        extract_outputs = [extract_btn, form_placeholder, form_container, fields_state, action_row, extraction_in_progress] + field_rows + text_inputs
        N = len(extract_outputs)
        def on_extract(img_path):
            def fill_outputs(updates):
                if len(updates) < N:
                    updates += [gr.update()] * (N - len(updates))
                return updates[:N]

            print(f"[on_extract] Called with img_path: {img_path}")
            if not img_path:
                print("[on_extract] No image path provided.")
                yield fill_outputs([
                    gr.update(value="Please upload an image first. | कृपया पहले छवि अपलोड करें।", interactive=True, variant="secondary"),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(visible=True),
                    False  # extraction_in_progress
                ])
                return
            # Set extraction_in_progress True
            yield fill_outputs([
                gr.update(value="Extracting... | निकाल रहे हैं...", interactive=False, variant="secondary", elem_classes=["skyblue-btn"]),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(),
                gr.update(visible=True),
                True  # extraction_in_progress
            ])
            try:
                fields = extract_fields_from_image(img_path)
                print(f"[on_extract] extract_fields_from_image returned: {fields}")
            except gr.Error as e:
                print(f"[on_extract] Exception during extraction: {e}")
                yield fill_outputs([
                    gr.update(value=str(e), interactive=True, variant="stop"),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(visible=True),
                    False  # extraction_in_progress
                ])
                return

            if not fields:
                print("[on_extract] No fields found after extraction.")
                yield fill_outputs([
                    gr.update(value="No fields found. | कोई फ़ील्ड नहीं मिली।", interactive=True, variant="stop"),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(),
                    gr.update(visible=True),
                    False  # extraction_in_progress
                ])
                return

            num_fields = min(len(fields), MAX_FIELDS)
            print(f"[on_extract] num_fields to show: {num_fields}")
            row_updates = [gr.update(visible=i < num_fields) for i in range(MAX_FIELDS)]
            text_updates = [gr.update(label=fields[i] if i < num_fields else "") for i in range(MAX_FIELDS)]
            result = [
                gr.update(value=f"Extracted! | निकाला गया!", interactive=False, variant="secondary", elem_classes=["skyblue-btn"]),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.State(fields),
                gr.update(visible=True),
                False  # extraction_in_progress
            ] + row_updates + text_updates
            print(f"[on_extract] Yielding result with {len(result)} outputs.")
            yield fill_outputs(result)

        def submit_form(*values):
            fields = values[-1]
            text_values = values[:MAX_FIELDS]
            # If fields is a gr.State, get its value
            if hasattr(fields, 'value'):
                fields = fields.value
            if not fields:
                return gr.update(value="Error: No fields to submit.", variant="stop", interactive=False), gr.update(visible=True)
            data = dict(zip(fields, text_values[:len(fields)]))
            try:
                save_to_csv(data)
                return gr.update(value="Submitted! | सबमिट हो गया", variant="success", interactive=False), gr.update(visible=True)
            except gr.Error as e:
                return gr.update(value=str(e), variant="stop", interactive=False), gr.update(visible=True)

        # --- Start New Form Logic ---
        def start_new_form():
            # Reset all form fields and UI state
            return [
                gr.update(value="", interactive=True, variant="secondary"),  # status box
                gr.update(visible=True),  # form_placeholder
                gr.update(visible=False),  # form_container
                gr.State([]),  # fields_state
                gr.update(visible=False),  # action_row
                False,  # extraction_in_progress
            ] + [gr.update(visible=False) for _ in range(MAX_FIELDS)] + [gr.update(value="", label="") for _ in range(MAX_FIELDS)]

        # --- Button/Callback Wiring ---
        sample_gallery.select(fn=on_sample_click, inputs=None, outputs=[image_input, image_uploaded])
        extract_btn.click(
            fn=on_extract,
            inputs=image_input,
            outputs=extract_outputs,
            show_progress="full",
            concurrency_limit=1
        )
        submit_btn.click(
            fn=submit_form,
            inputs=text_inputs + [fields_state],
            outputs=[submit_btn, action_row],
            show_progress="full",
            concurrency_limit=1
        )
        new_form_btn.click(
            fn=start_new_form,
            inputs=None,
            outputs=extract_outputs,
            concurrency_limit=1
        )
        # Optionally, add a button to clear cache
        # clear_cache_btn.click(fn=clear_cache_and_reset, inputs=None, outputs=status_box)

    return demo

# --- App Launch Block ---
if __name__ == "__main__":
    ui = main_ui()
    ui.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
