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
- **Translation Methods**: Google, Helsinki-NLP, Sogou, and Bing translators
- **Text Rendering**: Automatically fits translated text into speech bubbles

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

## Installation

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python app.py
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference