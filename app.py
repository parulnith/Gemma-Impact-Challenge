import sys
import platform
import re
import os

# Add current directory to Python path for utils module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
CACHE_FILE = "/tmp/image_cache.pkl"
AUDIO_CACHE_FILE = "/tmp/audio_cache.pkl"

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

# Legacy cache variables - now handled by modular CacheManager
image_cache = {}  # Kept for backward compatibility
audio_cache = {}  # Kept for backward compatibility

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

# Legacy support for backward compatibility
processor = None
model = None
device = torch.device("cpu")

def load_model_with_fallback():
    """Legacy function - now uses modular ModelHandler"""
    global processor, model
    
    # Use modular model handler
    success = model_handler.load_model()
    if success:
        model = model_handler.model
        processor = model_handler.processor
        return True
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
    """Legacy function - now uses modular ImageProcessor"""
    return image_processor.get_image_hash(image_path)

def get_audio_hash(audio_data):
    if isinstance(audio_data, tuple) and len(audio_data) == 2:
        sr, audio_np_array = audio_data
        return hashlib.md5(audio_np_array.tobytes() + str(sr).encode()).hexdigest()
    return None

def save_to_csv(data, csv_path="/tmp/forms.csv"):
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
# MODULAR AI PROCESSING 
# =============================================================================

# Import modular components
from utils import ImageProcessor, OutputParser, CacheManager, ModelHandler, get_logger

# Initialize modular components
image_processor = ImageProcessor(max_size=512)
output_parser = OutputParser()
cache_manager = CacheManager()
model_handler = ModelHandler()
logger = get_logger("ExtractionEngine")

def extract_fields_from_image(image_path, progress=None):
    """
    Main extraction function - Modular architecture with separated concerns
    
    This function orchestrates multiple specialized components:
    1. ImageProcessor: Handles image loading and preprocessing
    2. CacheManager: Manages intelligent caching for performance
    3. ModelHandler: Manages model loading and inference
    4. OutputParser: Parses AI responses using multiple strategies
    5. Logger: Provides consistent, professional logging
    
    Args:
        image_path (str): Path to uploaded form image
        progress (callable): Gradio progress callback for UI updates
        
    Returns:
        list: Clean list of extracted field names
        
    Architecture: Modular, testable, maintainable senior-level code
    """
    # Initialize extraction process
    logger.section_start("FORM_FIELD_EXTRACTION", 
                        f"Processing: {os.path.basename(image_path) if image_path else 'Unknown'}")
    
    if progress:
        progress(0.05, desc="Starting extraction pipeline...")
    
    # Step 1: Image preprocessing with modular component
    logger.step(1, "Image preprocessing", "starting")
    processed_image, image_hash, img_metadata = image_processor.preprocess(image_path)
    
    if processed_image is None:
        logger.error("Image preprocessing failed", img_metadata.get('error', 'Unknown error'))
        return []
    
    logger.success("Image preprocessed successfully", 
                  f"Size: {img_metadata['original_size']} -> {img_metadata['processed_size']}")
    
    if progress:
        progress(0.15, desc="Checking cache...")
    
    # Step 2: Smart caching check
    logger.step(2, "Cache lookup", "checking")
    cached_fields = cache_manager.get_image_extraction(image_hash)
    
    if cached_fields:
        logger.cache_event("hit", f"Found {len(cached_fields)} cached fields")
        if progress:
            for p in [0.3, 0.6, 0.9, 1.0]:
                progress(p, desc="Using cached results...")
                time.sleep(0.05)
        
        logger.section_end("FORM_FIELD_EXTRACTION", True, 
                          f"Returned {len(cached_fields)} cached fields")
        return cached_fields
    
    logger.cache_event("miss", "No cached results found, proceeding with AI extraction")
    
    if progress:
        progress(0.25, desc="Loading AI model...")
    
    # Step 3: Model loading and validation
    logger.step(3, "AI model initialization", "loading")
    if not model_handler.load_model():
        logger.error("Model loading failed")
        return ["[Error] Failed to load AI model"]
    
    logger.model_event("loaded", "Gemma 3n 4B ready for inference")
    
    if progress:
        progress(0.40, desc="Running AI inference...")
    
    # Step 4: AI inference with engineered prompt
    logger.step(4, "AI inference", "running")
    prompt = "<image_soft_token> Extract all form field labels from this image. Return ONLY a JSON object where keys are field names and values are 'text'. For Hindi/Devanagari forms, preserve the original script. Do NOT include any introductory or concluding text."
    
    try:
        ai_response, inference_metadata = model_handler.generate_response(
            processed_image, 
            prompt,
            generation_config={
                'max_new_tokens': 600,
                'do_sample': True,
                'num_beams': 3,
                'repetition_penalty': 1.2,
                'length_penalty': 1.0,
                'early_stopping': True,
            }

        )
        
        logger.performance("AI inference", inference_metadata['generation_time'],
                         {'total_tokens': 600, 'device': inference_metadata['device']})
        
    except Exception as e:
        logger.error("AI inference failed", str(e))
        return []
    
    if progress:
        progress(0.80, desc="Parsing AI response...")
    
    # Step 5: Intelligent output parsing
    logger.step(5, "Output parsing", "analyzing")
    text_result = ai_response
    import re, json

    def extract_all_json(text):
        jsons = []
        stack = []
        start = None
        for i, c in enumerate(text):
            if c == '{':
                if not stack:
                    start = i
                stack.append('{')
            elif c == '}':
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        candidate = text[start:i+1]
                        try:
                            obj = json.loads(candidate)
                            jsons.append(obj)
                        except Exception:
                            pass
                        start = None
        return jsons

    json_objs = extract_all_json(text_result)
    largest_json = None
    max_fields = 0
    if json_objs:
        for obj in json_objs:
            def flatten_json(json_obj, parent_key=''):
                flat = {}
                for key, value in json_obj.items():
                    if isinstance(value, dict):
                        if "type" in value:
                            flat[key] = value["type"]
                        else:
                            nested_flat = flatten_json(value, f"{parent_key}{key}_")
                            flat.update(nested_flat)
                    else:
                        flat[key] = value
                return flat
            flat = flatten_json(obj)
            if len(flat) > max_fields:
                largest_json = flat
                max_fields = len(flat)
        if largest_json:
            extracted_fields = list(largest_json.keys())
        else:
            extracted_fields = []
    else:
        # Fallback: extract lines with colon
        lines = [line.strip() for line in text_result.split('\n') if line.strip()]
        fields = []
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                field_name = parts[0].strip()
                if field_name and len(field_name) < 50:
                    fields.append(field_name)
        if not fields:
            for idx, line in enumerate(lines):
                if len(line) < 50:
                    fields.append(f"field_{idx+1}")
        extracted_fields = fields

    # Step 6: Cache successful results
    if extracted_fields and image_hash:
        logger.step(6, "Result caching", "saving")
        cache_manager.cache_image_extraction(
            image_hash,
            extracted_fields,
            {
                'extraction_method': "robust_json",
                'inference_time': inference_metadata['generation_time'],
                'image_metadata': img_metadata
            }
        )

    logger.section_end("FORM_FIELD_EXTRACTION", len(extracted_fields) > 0, f"Extracted {len(extracted_fields)} fields")
    if progress:
        progress(1.0, desc="Extraction complete")
    return extracted_fields

def transcribe_audio(audio_data, progress=gr.Progress()):
    """Modular audio transcription using specialized components"""
    audio_logger = get_logger("AudioTranscription")
    
    if progress:
        progress(0.05, desc="Starting audio transcription...")
    
    # Step 1: Validate input
    if audio_data is None:
        audio_logger.error("No audio data provided")
        return "[Error] No audio provided."
    
    # Step 2: Check cache
    audio_hash = get_audio_hash(audio_data)
    if audio_hash:
        cached_result = cache_manager.get_audio_transcription(audio_hash)
        if cached_result:
            audio_logger.cache_event("hit", f"Found cached transcription")
            if progress:
                progress(1.0, desc="Using cached transcription!")
            return cached_result
    
    audio_logger.cache_event("miss", "No cached transcription found")
    
    if progress:
        progress(0.2, desc="Loading AI model...")
    
    # Step 3: Ensure model is loaded
    if not model_handler.load_model():
        audio_logger.error("Failed to load model for transcription")
        return "[Error] Failed to load AI model."
    
    if progress:
        progress(0.4, desc="Processing audio...")
    
    # Step 4: Process audio data
    try:
        if isinstance(audio_data, str):
            audio_np_array, sr = sf.read(audio_data)
        elif isinstance(audio_data, tuple) and len(audio_data) == 2:
            sr, audio_np_array = audio_data
        else:
            audio_logger.error("Invalid audio format")
            return "[Error] Invalid audio input format."
        
        if audio_np_array is None or len(audio_np_array) == 0:
            audio_logger.error("Empty audio data")
            return "[Error] Empty audio data."
        
        # Audio preprocessing
        if audio_np_array.ndim > 1:
            audio_np_array = np.mean(audio_np_array, axis=1)
        
        if np.max(np.abs(audio_np_array)) > 0:
            audio_np_array = audio_np_array / (np.max(np.abs(audio_np_array)) + 1e-9)
        
        # Resample if needed
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
        
        audio_logger.success("Audio preprocessing completed", f"Sample rate: {sr}, Length: {len(audio_np_array)}")
        
    except Exception as e:
        audio_logger.error("Audio preprocessing failed", str(e))
        return "[Error] Audio processing failed."
    
    if progress:
        progress(0.6, desc="Running AI transcription...")
    
    # Step 5: Run transcription using modular model handler
    try:
        transcription, trans_metadata = model_handler.transcribe_audio(
            (sr, audio_np_array),
            "Transcribe this audio in Hindi (Devanagari script)."
        )
        
        audio_logger.performance("Audio transcription", trans_metadata['generation_time'])
        
    except Exception as e:
        audio_logger.error("Transcription failed", str(e))
        return "[Error] Transcription failed."
    
    if progress:
        progress(0.9, desc="Caching results...")
    
    # Step 6: Cache successful transcription
    if audio_hash and transcription and transcription != "Audio not clear.":
        cache_success = cache_manager.cache_audio_transcription(
            audio_hash, 
            transcription,
            {'transcription_time': trans_metadata['generation_time']}
        )
        
        if cache_success:
            audio_logger.cache_event("save", "Transcription cached for future use")
    
    if progress:
        progress(1.0, desc="Transcription complete!")
    
    audio_logger.success("Transcription completed", f"Result: {transcription[:50]}...")
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
# ASHA Form Digitizer & Hindi Voice Transcriber

**Digitizing rural health, empowering ASHA workers with AI.**

<p style='font-size:1.08rem; color:#263238; margin-bottom:1.2em;'>
This application is designed to help <b>ASHA (Accredited Social Health Activist) workers</b>—the backbone of India's rural healthcare system—quickly digitize handwritten forms and transcribe Hindi voice input. ASHA workers are often the first point of contact for healthcare in villages, but their work is slowed by manual paperwork and language barriers. This tool streamlines their workflow, making data entry faster, more accurate, and accessible even for those more comfortable with Hindi speech than typing.
</p>
<ul class='asha-list'>
    <li><b>Image-based field extraction:</b> Upload a photo of an ASHA form and the app will automatically detect and extract all field labels, ready for digital entry.</li>
    <li><b>Hindi voice transcription:</b> Fill any field by speaking in Hindi (Devanagari script) for instant, accurate transcription.</li>
    <li><b>Data export:</b> All submitted data is saved in a CSV for further use or analysis.</li>
</ul>
<div class='asha-note'>
    <b>Why this matters:</b> ASHA workers serve over 900 million people in rural India, often with limited digital literacy and resources. By making form digitization and voice transcription seamless, this app saves time, reduces errors, and helps bring rural health data into the digital age—empowering both workers and the communities they serve.
</div>
<div class='asha-note' style='color:#1a237e; font-size:1.05rem; margin-top:0.5em;'>
    <b>Note</b> While you are reading this, the <b>Gemma 3n model</b> is being loaded in the background to ensure a smooth and fast demo experience. Please explore each step—the workflow is strictly gated for demo clarity. All features are designed for real-world usability and hackathon evaluation.
</div>
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