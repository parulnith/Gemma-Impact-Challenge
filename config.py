import os
import pickle
import time
import pandas as pd
from dotenv import load_dotenv

gemma_model_id = "google/gemma-3n-E4B-it"
cache_file = "image_cache.pkl"
audio_cache_file = "audio_cache.pkl"

# --- Cache Loading ---
def load_cache(file_path):
    """Loads a cache file if it exists, otherwise returns an empty dictionary."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[Cache] Could not load cache from {file_path}: {e}")
    return {}

image_cache = load_cache(cache_file)
audio_cache = load_cache(audio_cache_file)

# --- Hugging Face Token ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

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
        print(f"[CSV] Data for {len(rows_to_save)} fields saved to {csv_path}")
    except Exception as e:
        raise Exception(f"Failed to save to CSV: {e}")
