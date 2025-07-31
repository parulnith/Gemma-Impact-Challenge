# On Device smart ASHA Forms(Gemma 3n Powered)

**Empowering ASHA workers with AI for seamless form digitization and Hindi voice transcription**

[![Made with Gemma 3n](https://img.shields.io/badge/Made%20with-Gemma%203n%204B-blue)](https://huggingface.co/google/gemma-3n-E4B-it)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🌟 Project Overview

This project leverages **Google's Gemma 3n 4B model** to create an AI-powered solution that helps [ASHA](https://nhm.gov.in/index1.php?lang=1&level=1&sublinkid=150&lid=226) - **Accredited Social Health Activist** workers in rural India digitize forms and transcribe Hindi voice input. The application  addresses critical challenges in rural healthcare data management by making form processing faster, more accurate, and accessible even for those with limited digital literacy.

## ✨ Highlights & Gemma 3n Capabilities

**This project is a showcase of Gemma 3n's advanced multimodal AI capabilities, designed for real-world social impact:**

1. **Multimodal AI**: Combines vision (image understanding), audio (speech-to-text), and language (text processing) in a single workflow using Gemma 3n 4B.
2. **Vision Intelligence**: Uses Gemma 3n's vision model to detect and extract handwritten form fields from images, enabling digitization of paper forms.
3. **Audio & Speech**: Leverages Gemma 3n's audio pipeline for accurate Hindi voice transcription, converting spoken input to Devanagari script in real time.
4. **Multilingual Support**: Gemma 3n enables robust form field recognition and voice transcription in Hindi and other languages, making the tool accessible to diverse users.
5. **On-Device, Private AI**: All inference runs locally after model download—no internet required, ensuring privacy and accessibility in rural settings.
6. **Real-World Impact**: Directly addresses challenges faced by ASHA workers, healthcare admins, and researchers in rural India.
7. **Scalable, Modular Design**: Built for easy extension and adaptation to new forms, languages, and use cases.

> **Gemma 3n powers the entire pipeline: from image to structured data, from voice to text, and across languages.**

---

**🔗 Submission Links for Judges:**


---

**🔗 Submission Links:**

- 📄 **Writeup:** [View Writeup](#) <!-- Replace # with your writeup link -->
- 🎥 **Demo Video:** [Watch Video](#) <!-- Replace # with your video link -->
- 🚀 **Live Demo:** [Try the Application](https://huggingface.co/spaces/ParulPandey/demo)

---
### 🎯 Key Features
![](images/Architecture.png)

- **🔍 Intelligent Form Field Extraction**: Upload photos of forms and automatically extract all field labels using Gemma 3n's vision capabilities
- **🎤 Hindi Voice Transcription**: Fill form fields by speaking in Hindi - the AI provides instant, accurate Devanagari script transcription
- **💻 Fully On-Device Processing**: Runs completely offline once the Gemma 3n model is downloaded - no internet required for inference
- **💾 Smart Caching System**: Reduces processing time by intelligently caching extraction results
- **📊 CSV Export**: All submitted data is automatically saved in structured CSV format for analysis
- **🏗️ Modular Architecture**: Clean, maintainable code with separated concerns for scalability

### 🔧 Usage

### Step-by-Step Workflow

1. **About & Demo Info**: Learn about the application and its impact
2. **Upload Image**: Upload a photo of your handwritten form or try sample images
3. **Extract Fields**: Click "Extract Fields" to automatically detect all form labels
4. **Fill Form**: 
   - Type values directly or
   - Use the microphone icon to speak in Hindi for automatic transcription
5. **Submit**: Save your data to CSV for further processing

### Supported Languages

- **Primary**: Hindi (Devanagari script)
- **Interface**: English and Hindi bilingual support
- **Form Recognition**: Multi-language form field detection


## 📂 Project Structure & File Roles

Here's how the main files and modules work together:

- **app.py**: The main entry point. Launches the Gradio UI, handles user interactions, and coordinates all processing.
- **utils/model_handler.py**: Loads and manages the Gemma 3n model. Handles all inference requests from `app.py` (for both vision and audio tasks).
- **utils/image_processor.py**: Preprocesses uploaded images (resizing, normalization, etc.) before sending them to the model for field extraction.
- **utils/cache_manager.py**: Implements smart caching. Stores and retrieves results of previous extractions to speed up repeated tasks.
- **utils/output_parser.py**: Parses and structures the raw output from the model into usable form fields and values for the UI.
- **utils/logger.py**: Handles logging of key events, errors, and user actions for debugging and monitoring.
- **samples/**: Contains sample form images for demo/testing.
- **requirements.txt**: Lists all Python dependencies.

**How they interact:**

1. `app.py` receives an image or audio input from the user via the Gradio UI.
2. The image is sent to `image_processor.py` for preprocessing.
3. Preprocessed data is passed to `model_handler.py`, which runs inference using Gemma 3n.
4. The output is parsed by `output_parser.py` to extract structured fields/values.
5. Results are cached by `cache_manager.py` for faster future access.
6. All major actions and errors are logged by `logger.py`.

This modular design keeps the codebase clean, maintainable, and easy to extend.


## 🏥 Impact & Use Cases

- **Patient Registration**: Digitize handwritten patient intake forms

## 🚀 Performance Optimizations
- **Memory Management**: Intelligent model loading and unloading
- **Progressive UI**: Real-time progress indicators for better UX



2. **Real-World Problem Solving**: Addresses actual challenges faced by ASHA workers
3. **Accessibility Focus**: Hindi voice input for better user adoption
4. **Performance Optimization**: CPU-friendly implementation for resource constraints











## 📋 Installation & Setup

### Prerequisites

- Python 3.11 or higher
- Hugging Face account and token
1. **Clone the repository**
   ```bash
   git clone [your-repo-url]

2. **Install dependencies**
   ```bash
   ```bash
   # Create .env file
   echo "HF_TOKEN=your_hugging_face_token_here" > .env
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the interface**
   - Open your browser to `http://localhost:7860`
   - Follow the guided workflow: Upload → Extract → Fill → Submit

### Environment Variables

Create a `.env` file with:
```env
HF_TOKEN=your_hugging_face_token_here
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

