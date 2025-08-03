"""
ASHA Form Digitization Application - Main Entry Point

This is the main entry point for the ASHA (Accredited Social Health Activist) 
Form Digitization application. The app helps rural healthcare workers digitize 
handwritten forms using AI-powered field extraction and Hindi voice transcription.

Features:
- Image-based form field extraction using Google's Gemma 3n 4B model
- Hindi voice transcription for easy data entry
- Professional UI with tab-based workflow
- CSV export functionality for data analysis

Usage:
    python main.py

The application will start a Gradio web interface accessible at:
http://localhost:7860

Author: Gemma Impact Challenge Team
License: MIT
"""

from ui import main_ui

if __name__ == "__main__":
    # Initialize and launch the Gradio UI
    ui = main_ui()
    ui.queue().launch(
        server_name="0.0.0.0",  # Allow access from any IP
        server_port=7860,       # Standard port for Gradio apps
        share=False             # Set to True for public sharing via ngrok
    )
