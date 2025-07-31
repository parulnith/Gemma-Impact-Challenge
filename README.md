# ASHA Form Digitizer & Hindi Voice Transcriber

**Empowering ASHA workers with AI for seamless form digitization and Hindi voice transcription**

[![Made with Gemma 3n](https://img.shields.io/badge/Made%20with-Gemma%203n%204B-blue)](https://huggingface.co/google/gemma-3n-E4B-it)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🌟 Project Overview

This project leverages **Google's Gemma 3n 4B model** to create an AI-powered solution that helps [ASHA](https://nhm.gov.in/index1.php?lang=1&level=1&sublinkid=150&lid=226) (Accredited Social Health Activist) workers in rural India digitize handwritten forms and transcribe Hindi voice input. The application addresses critical challenges in rural healthcare data management by making form processing faster, more accurate, and accessible even for those with limited digital literacy.

### 🎯 Key Features
![](images/Architecture.png)

- **🔍 Intelligent Form Field Extraction**: Upload photos of handwritten forms and automatically extract all field labels using Gemma 3n's vision capabilities
- **🎤 Hindi Voice Transcription**: Fill form fields by speaking in Hindi - the AI provides instant, accurate Devanagari script transcription
- **💻 Fully On-Device Processing**: Runs completely offline once the Gemma 3n model is downloaded - no internet required for inference
- **💾 Smart Caching System**: Reduces processing time by intelligently caching extraction results
- **📊 CSV Export**: All submitted data is automatically saved in structured CSV format for analysis
- **🏗️ Modular Architecture**: Clean, maintainable code with separated concerns for scalability

## 🚀 Demo

Try the live application: [Demo](https://huggingface.co/spaces/ParulPandey/demo)

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


## 🛠️ Technical Implementation

### Core Technologies

- **AI Model**: [Google Gemma 3n 4B](https://huggingface.co/google/gemma-3n-E4B-it) - Multimodal model for vision and language tasks
- **Framework**: Python 3.11+ with PyTorch
- **UI**: Gradio for web interface

### Architecture

```
app.py                    # Main Gradio application
├── utils/
│   ├── model_handler.py     # Gemma 3n model management
│   ├── image_processor.py   # Image preprocessing pipeline
│   ├── cache_manager.py     # Intelligent caching system
│   ├── output_parser.py     # AI response parsing
│   └── logger.py           # Professional logging
├── samples/                 # Demo form images
└── requirements.txt        # Dependencies
```

### Key Components

1. **ModelHandler**: Manages Gemma 3n model loading, inference, and optimization
2. **ImageProcessor**: Handles image preprocessing with intelligent resizing
3. **CacheManager**: Implements smart caching for performance optimization
4. **OutputParser**: Robust parsing of AI responses with fallback strategies




## 🧠 Gemma 3n Integration

This project showcases use of **Google's Gemma 3n 4B model**:

### Vision Tasks
- Form field detection and extraction

### Language Tasks 
-  
- Hindi voice-to-text transcription
- Multilingual 


## 🏥 Impact & Use Cases

### Primary Users
- **ASHA Workers**: Frontline health workers in rural India
- **Healthcare Administrators**: Data collection and analysis teams
- **Public Health Researchers**: Community health data analysts

### Real-World Applications
- **Patient Registration**: Digitize handwritten patient intake forms
- **Health Surveys**: Convert paper-based community health assessments
- **Vaccination Records**: Streamline immunization data entry
- **Medical History**: Digitize patient medical record forms


## 🚀 Performance Optimizations

- **Smart Caching**: Avoids reprocessing identical forms
- **CPU Optimization**: Efficient inference on resource-constrained devices
- **Memory Management**: Intelligent model loading and unloading
- **Progressive UI**: Real-time progress indicators for better UX



### Highlights

1. **Multimodal AI Application**: Combines vision and language capabilities of Gemma 3n
2. **Real-World Problem Solving**: Addresses actual challenges faced by ASHA workers
3. **Accessibility Focus**: Hindi voice input for better user adoption
4. **Performance Optimization**: CPU-friendly implementation for resource constraints
5. **Scalable Architecture**: Modular design enables easy feature expansion













## 📋 Installation & Setup

### Prerequisites

- Python 3.11 or higher
- Hugging Face account and token

### Quick Start

1. **Clone the repository**
   ```bash
   git clone [your-repo-url]
   cd asha-form-digitizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Hugging Face token**
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

