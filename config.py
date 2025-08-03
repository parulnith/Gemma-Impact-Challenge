"""
Configuration and Utility Functions for ASHA Form Application

This module contains configuration settings, utility functions, and data persistence
logic for the ASHA form digitization application.

Key Components:
- Model configuration (Gemma 3n 4B settings)
- Cache management for images and audio
- CSV export functionality for form data
- File I/O utilities

Dependencies:
- pandas: For CSV data manipulation
- pickle: For cache serialization
- dotenv: For environment variable management
"""

import os
import pickle
import time
import pandas as pd
from dotenv import load_dotenv

# Model Configuration
gemma_model_id = "google/gemma-3n-E4B-it"  # Google's Gemma 3n 4B model for on-device AI

# Cache Configuration
cache_file = "image_cache.pkl"      # Cache for processed image results
audio_cache_file = "audio_cache.pkl"  # Cache for audio transcription results

# --- Cache Loading ---
def load_cache(file_path):
    """
    Load cache data from pickle file.
    
    Args:
        file_path (str): Path to the cache file
        
    Returns:
        dict: Cached data or empty dict if file doesn't exist
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[Cache] Could not load cache from {file_path}: {e}")
    return {}

# Initialize cache dictionaries
image_cache = load_cache(cache_file)       # Image processing results cache
audio_cache = load_cache(audio_cache_file)  # Audio transcription results cache

# --- Hugging Face Token ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")  # Authentication token for Hugging Face API

def save_to_csv(data, csv_path="output/forms.csv"):
    """
    Save form data to CSV file in long format for easy analysis.
    
    This function takes a dictionary of form field names and values,
    converts them to a long format DataFrame with timestamps, and
    appends to a CSV file for persistent storage and analysis.
    
    Args:
        data (dict): Dictionary where keys are field names and values are field values
        csv_path (str): Path to the output CSV file (default: "output/forms.csv")
        
    Raises:
        Exception: If file creation or writing fails
        
    Example:
        data = {"Name": "John Doe", "Age": "25", "Village": "Rampur"}
        save_to_csv(data)
        # Creates CSV with columns: Timestamp, Field_Name, Field_Value
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Convert dictionary to long format with timestamps
        rows_to_save = [
            {
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), 
                "Field_Name": field, 
                "Field_Value": value
            }
            for field, value in data.items()
        ]
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows_to_save)
        header = not os.path.exists(csv_path)  # Only add header if file doesn't exist
        df.to_csv(csv_path, mode="a", header=header, index=False)
        
        print(f"[CSV] Data for {len(rows_to_save)} fields saved to {csv_path}")
    except Exception as e:
        raise Exception(f"Failed to save to CSV: {e}")
