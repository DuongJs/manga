---
title: MangaTranslator
emoji: 📚
colorFrom: yellow
colorTo: indigo
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
license: mit
---

# Manga Translator

A manga translation application that detects text bubbles, extracts text using OCR, and translates it to English.

## Features

- **Text Detection**: Uses YOLOv8 to detect speech bubbles in manga images
- **OCR Options**: 
  - **Manga OCR**: Specialized OCR for manga text
  - **PaddleOCR**: Powerful open-source OCR supporting 100+ languages with CPU optimization
- **Translation Methods**: 
  - Google Translator
  - Helsinki-NLP (HuggingFace model)
  - Sogou Translate
  - Bing Translator
  - **Gemini API**: Advanced translation with built-in OCR in a single API call
- **Text Rendering**: Automatically fits translated text into speech bubbles
- **Batch Processing**: Process multiple images simultaneously

## PaddleOCR Integration

PaddleOCR is an open-source OCR tool built on PaddlePaddle deep learning framework. It supports over 100 languages and provides high-accuracy text detection and recognition.

### Features of PaddleOCR

- **CPU Optimized**: Runs efficiently on CPU without requiring GPU
- **High Accuracy**: State-of-the-art text detection and recognition models
- **Angle Classification**: Automatically handles rotated text
- **Multi-language Support**: Supports Japanese and many other languages

### Usage

1. Select "PaddleOCR" from the OCR Method dropdown in the interface
2. Upload your manga image
3. Choose translation method and font
4. Click Submit to process

### Configuration

PaddleOCR is configured with the following settings for optimal CPU performance:

- `use_gpu=False`: CPU-only processing
- `use_angle_cls=True`: Handles rotated text
- `lang='japan'`: Optimized for Japanese text recognition
- `show_log=False`: Minimal logging for cleaner output

### Performance Tips

- PaddleOCR model is loaded once and reused for efficiency
- Uses MKL-DNN optimization when available on x86 CPUs
- Processes images in their original size for best accuracy

## Gemini API Integration

The application now supports Google's Gemini API for advanced translation with built-in OCR capabilities.

### Features of Gemini Translation

- **Combined OCR + Translation**: Performs OCR and translation in a single API call using Gemini's multimodal capabilities
- **High Accuracy**: Leverages Gemini 2.0 Flash model for superior text recognition and translation
- **Fast Processing**: Uses optimized configuration with thinking budget set to 0 for faster responses
- **No Separate OCR Required**: Gemini processes the image directly, eliminating the need for separate OCR step

### Setup

1. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the API key as an environment variable:
   ```bash
   export GEMINI_API_KEY='your-api-key-here'
   ```
   Or enter it directly in the web interface

### Usage

1. Select "Gemini (OCR + Translation)" from the Translation Method dropdown
2. Enter your Gemini API key (if not set as environment variable)
3. Upload your manga image
4. Click Submit to process

### How It Works

The Gemini integration uses multimodal input capabilities to:
1. Send the manga panel image directly to Gemini API
2. Gemini extracts Japanese text using its built-in OCR
3. Gemini translates the extracted text to English in the same API call
4. The translated text is rendered back into the speech bubble

This approach is more efficient than traditional OCR + translation pipelines and often provides better results for manga text.

### API Configuration

The Gemini translator is configured with:
- Model: `gemini-2.0-flash-exp` (experimental Flash model)
- `thinkingBudget: 0` - Disables extended thinking mode for faster responses
- `temperature: 0.3` - Lower temperature for more consistent translations
- System instruction optimized for manga translation

## Batch Processing

The application supports processing multiple manga images at once:
- Upload multiple images
- All images will be processed with the same translation settings
- Results are returned as a list of translated images

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python app.py
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference