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

A manga translation application that detects text bubbles, extracts text using OCR, and translates it to Vietnamese (tiếng Việt).

## Features

- **Text Detection**: Uses YOLOv8 to detect speech bubbles in manga images with improved duplicate filtering
- **OCR Options**: 
  - **Manga OCR**: Specialized OCR for manga text
  - **PaddleOCR**: Powerful open-source OCR supporting 100+ languages with CPU optimization
- **Translation Methods**: 
  - Google Translator
  - Helsinki-NLP (HuggingFace model)
  - Sogou Translate
  - Bing Translator
  - **Gemini API**: Advanced translation with built-in OCR in a single API call
- **Vietnamese Translation**: All translations default to Vietnamese (tiếng Việt)
- **Text Rendering**: Automatically fits translated text into speech bubbles
- **API Key Management**: Store and rotate multiple Gemini API keys automatically
- **Custom Prompts**: Customize translation style and context for Gemini translations
- **🆕 Batch Translation**: Translate multiple manga pages at once with optimized performance
  - Upload multiple images simultaneously
  - **Gemini Batch API Optimization**: Intelligent chunking processes ~20 images per API request
  - Processes 100 images in just 5 API calls instead of 100 (20x efficiency!)
  - Real-time progress tracking
  - Download all results as ZIP
  - Comprehensive statistics
  - See [BATCH_TRANSLATE_GUIDE.md](BATCH_TRANSLATE_GUIDE.md) for details

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

PaddleOCR is configured with the following default settings for optimal CPU performance:

- `device='cpu'`: CPU-only processing
- `use_textline_orientation=True`: Handles rotated text detection and correction
- `lang='japan'`: Optimized for Japanese text recognition

### Performance Tips

- PaddleOCR model is loaded once and reused for efficiency
- Uses MKL-DNN optimization when available on x86 CPUs
- Processes images in their original size for best accuracy

## Gemini API Integration

The application now supports Google's Gemini API for advanced translation with built-in OCR capabilities.

### Features of Gemini Translation

- **Combined OCR + Translation**: Performs OCR and translation in a single API call using Gemini's multimodal capabilities
- **Intelligent Batch Processing**: Automatically chunks large batches into optimal sizes (~20 images per API request)
- **High Efficiency**: Process 100 images with only 5 API requests instead of 100 individual calls
- **High Accuracy**: Leverages Gemini 2.5 Flash Lite model for superior text recognition and translation
- **Fast Processing**: Uses optimized configuration with thinking budget set to 0 for faster responses
- **No Separate OCR Required**: Gemini processes the image directly, eliminating the need for separate OCR step
- **Custom Prompts**: Add custom instructions to control translation style and context
- **API Key Rotation**: Automatically rotates through multiple API keys for high-volume usage

### Setup

1. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Configure your API key using one of these methods:
   - Add keys through the "API Key Management" tab in the web interface (recommended)
   - Create an `api_keys.json` file (see `api_keys.json.example`)
   - Set the `GEMINI_API_KEY` environment variable
   - Enter the key directly in the UI when translating

For detailed setup instructions, see [API_KEY_SETUP.md](API_KEY_SETUP.md)

### Usage

1. Select "Gemini (OCR + Translation)" from the Translation Method dropdown
2. Enter your Gemini API key (if not using the API Key Manager)
3. Optionally add a custom prompt (e.g., "Use casual Vietnamese style")
4. Upload your manga image
5. Adjust detection thresholds if needed
6. Click Translate to process

### API Key Management

The application includes a built-in API Key Manager that:
- Stores multiple API keys securely in `api_keys.json`
- Automatically rotates keys in round-robin fashion
- Tracks usage statistics for each key
- Allows adding/removing keys through the web interface

To manually create the `api_keys.json` file, use the following format (see `api_keys.json.example`):
```json
{
  "api_keys": [
    {
      "key": "YOUR_GEMINI_API_KEY_HERE",
      "name": "Primary Key",
      "added_at": "2024-10-18T00:00:00.000000",
      "usage_count": 0
    }
  ],
  "current_index": 0,
  "last_updated": "2024-10-18T00:00:00.000000"
}
```

### Custom Prompts

Custom prompts allow you to control the translation style:
- "Use casual/informal Vietnamese style"
- "Translate in a formal tone"
- "Keep the original tone and emotion"
- "Use modern Vietnamese slang"

### How It Works

The Gemini integration uses multimodal input capabilities to:
1. Send the manga panel image directly to Gemini API
2. Gemini extracts Japanese text using its built-in OCR
3. Gemini translates the extracted text to Vietnamese in the same API call
4. The translated text is rendered back into the speech bubbles

This approach is more efficient than traditional OCR + translation pipelines and often provides better results for manga text.

### API Configuration

The Gemini translator is configured with:
- Model: `gemini-2.5-flash-lite` (lightweight Flash model)
- `thinkingBudget: 0` - Disables extended thinking mode for faster responses
- `temperature: 0.3` - Lower temperature for more consistent translations
- System instruction optimized for manga translation to Vietnamese

## Bubble Detection

The application uses improved bubble detection with:
- Configurable confidence threshold (default: 0.5)
- IoU-based duplicate filtering (default: 0.3)
- Additional overlap detection to prevent duplicate bubbles
- Adjustable parameters through the web interface

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python app.py
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference