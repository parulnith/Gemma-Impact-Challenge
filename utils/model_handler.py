"""
Model Handler Module
Manages Gemma 3n 4B model loading, inference, and optimization.
"""

import os
import sys
import time
import torch
import warnings
import requests
from typing import Dict, Any, Optional, Tuple
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image


class ModelHandler:
    """Handles Gemma 3n 4B model operations with intelligent loading and inference"""
    
    def __init__(self, model_id: str = "google/gemma-3n-E4B-it", device: str = "cpu"):
        """
        Initialize model handler
        
        Args:
            model_id (str): Hugging Face model identifier
            device (str): Device to run model on ('cpu' or 'cuda')
        """
        self.model_id = model_id
        self.device = torch.device(device)
        self.processor = None
        self.model = None
        self.is_loaded = False
        
        # Get HF token from environment
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            warnings.warn("HF_TOKEN not found in environment. May cause issues if model isn't cached.")
    
    def _check_network_connectivity(self) -> bool:
        """
        Check if Hugging Face servers are accessible
        
        Returns:
            bool: True if network is available
        """
        try:
            requests.get("https://huggingface.co", timeout=5)
            return True
        except requests.ConnectionError:
            return False
    
    def _get_model_cache_info(self) -> Dict[str, Any]:
        """
        Get information about model cache status
        
        Returns:
            dict: Cache information
        """
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_path = os.path.join(cache_dir, f"models--{self.model_id.replace('/', '--')}")
        
        return {
            'cache_dir': cache_dir,
            'model_cache_path': model_cache_path,
            'is_cached': os.path.exists(model_cache_path),
            'has_network': self._check_network_connectivity()
        }
    
    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the Gemma model and processor with offline fallback
        
        Args:
            force_reload (bool): Force reload even if already loaded
            
        Returns:
            bool: True if loaded successfully
        """
        if self.is_loaded and not force_reload:
            print("[ModelHandler] Model already loaded")
            return True
        
        print(f"[ModelHandler] Loading {self.model_id} on {self.device}...")
        
        # Get cache information
        cache_info = self._get_model_cache_info()
        use_offline = cache_info['is_cached'] and not cache_info['has_network']
        
        if use_offline:
            print("[ModelHandler] Network unavailable, using offline mode with cached model")
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        try:
            # Load processor
            print("[ModelHandler] Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                token=self.hf_token,
                local_files_only=use_offline
            )
            
            # Load model
            print("[ModelHandler] Loading model...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                token=self.hf_token,
                local_files_only=use_offline
            ).to(self.device).eval()
            
            self.is_loaded = True
            print(f"[ModelHandler] Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"[ModelHandler] CRITICAL: Model loading failed: {e}", file=sys.stderr)
            self.is_loaded = False
            return False
    
    def generate_response(self, image: Image.Image, prompt: str, 
                         generation_config: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response from image and prompt
        
        Args:
            image (PIL.Image): Input image
            prompt (str): Text prompt
            generation_config (dict, optional): Generation parameters
            
        Returns:
            tuple: (generated_text, generation_metadata)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Default generation config optimized for CPU
        default_config = {
            'max_new_tokens': 128,
            'do_sample': False,
            'num_beams': 1,
            'temperature': 0.1,
            'repetition_penalty': 1.1,
            'length_penalty': 1.0,
            'early_stopping': True,
            'pad_token_id': self.processor.tokenizer.eos_token_id
        }
        
        if generation_config:
            default_config.update(generation_config)
        
        # Tokenize inputs
        print("[ModelHandler] Tokenizing inputs...")
        tokenize_start = time.time()
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        tokenize_time = time.time() - tokenize_start
        
        # Generate response
        print(f"[ModelHandler] Running inference with config: {default_config}")
        generation_start = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **default_config)
        
        generation_time = time.time() - generation_start
        
        # Decode output
        decode_start = time.time()
        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        decode_time = time.time() - decode_start
        
        # Prepare metadata
        metadata = {
            'tokenize_time': tokenize_time,
            'generation_time': generation_time,
            'decode_time': decode_time,
            'total_time': tokenize_time + generation_time + decode_time,
            'generation_config': default_config,
            'input_shapes': {k: v.shape for k, v in inputs.items() if hasattr(v, 'shape')},
            'device': str(self.device)
        }
        
        return text, metadata
    
    def transcribe_audio(self, audio_data: tuple, prompt: str = None,
                        generation_config: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Transcribe audio using Gemma model
        
        Args:
            audio_data (tuple): (sample_rate, audio_array)
            prompt (str, optional): Custom prompt for transcription
            generation_config (dict, optional): Generation parameters
            
        Returns:
            tuple: (transcription, metadata)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if prompt is None:
            prompt = "Transcribe this audio in Hindi (Devanagari script)."
        
        # Default config for audio transcription
        default_config = {
            'max_new_tokens': 32,
            'do_sample': False,
            'num_beams': 1
        }
        
        if generation_config:
            default_config.update(generation_config)
        
        sr, audio_array = audio_data
        
        # Prepare messages for audio input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template and tokenize
        print("[ModelHandler] Processing audio input...")
        tokenize_start = time.time()
        input_dict = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            sampling_rate=sr
        )
        inputs = {k: v.to(self.device) for k, v in input_dict.items()}
        tokenize_time = time.time() - tokenize_start
        
        # Generate transcription
        generation_start = time.time()
        with torch.inference_mode():
            predicted_ids = self.model.generate(**inputs, **default_config)
        generation_time = time.time() - generation_start
        
        # Decode transcription
        decode_start = time.time()
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()
        decode_time = time.time() - decode_start
        
        # Post-process transcription
        transcription = self._clean_transcription(transcription, prompt)
        
        metadata = {
            'tokenize_time': tokenize_time,
            'generation_time': generation_time,
            'decode_time': decode_time,
            'total_time': tokenize_time + generation_time + decode_time,
            'sample_rate': sr,
            'audio_length': len(audio_array),
            'generation_config': default_config
        }
        
        return transcription, metadata
    
    def _clean_transcription(self, transcription: str, prompt: str) -> str:
        """
        Clean transcription output by removing artifacts
        
        Args:
            transcription (str): Raw transcription
            prompt (str): Original prompt
            
        Returns:
            str: Cleaned transcription
        """
        import re
        
        # Remove user/model tokens
        transcription = re.sub(r'\b(user|model)\s*[:：\-]+', '', transcription, flags=re.IGNORECASE)
        transcription = re.sub(r'\b(user|model)\b', '', transcription, flags=re.IGNORECASE)
        
        # Remove prompt text
        transcription = transcription.replace(prompt, '')
        
        # Clean whitespace and punctuation
        transcription = transcription.strip(' :\n\t-')
        
        # Return default if empty
        if not transcription or transcription.strip() == "":
            transcription = "Audio not clear."
        
        return transcription
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and status
        
        Returns:
            dict: Model information
        """
        cache_info = self._get_model_cache_info()
        
        return {
            'model_id': self.model_id,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'has_processor': self.processor is not None,
            'has_model': self.model is not None,
            'cache_info': cache_info,
            'hf_token_available': self.hf_token is not None
        }
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self.is_loaded = False
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[ModelHandler] Model unloaded and memory cleared")