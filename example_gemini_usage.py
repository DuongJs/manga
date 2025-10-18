#!/usr/bin/env python3
"""
Example script demonstrating Gemini API integration for manga translation.

This script shows how to use the GeminiTranslator class to:
1. Perform OCR and translation in a single API call
2. Process multiple images in batch

Usage:
    export GEMINI_API_KEY='your-api-key-here'
    python example_gemini_usage.py
"""

import os
from PIL import Image
from gemini_translator import GeminiTranslator


def example_single_image():
    """Example: Translate a single manga panel."""
    print("Example 1: Single Image Translation with Gemini")
    print("=" * 60)
    
    # Initialize translator with API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠ GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY='your-api-key'")
        return
    
    translator = GeminiTranslator(api_key=api_key)
    
    # Load a manga image (you would use your actual manga panel here)
    # For demonstration purposes, we'll just show the structure
    try:
        # image = Image.open('path/to/manga/panel.png')
        # translated_text = translator.ocr_and_translate(image)
        # print(f"Translated text: {translated_text}")
        print("Structure:")
        print("  translator = GeminiTranslator(api_key=api_key)")
        print("  image = Image.open('manga_panel.png')")
        print("  translated_text = translator.ocr_and_translate(image)")
        print("✓ Ready to use with actual images")
    except Exception as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Example: Process multiple manga panels in batch."""
    print("\nExample 2: Batch Processing with Gemini")
    print("=" * 60)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠ GEMINI_API_KEY environment variable not set")
        return
    
    translator = GeminiTranslator(api_key=api_key)
    
    # Load multiple manga images
    try:
        # images = [
        #     Image.open('panel1.png'),
        #     Image.open('panel2.png'),
        #     Image.open('panel3.png'),
        # ]
        # translated_texts = translator.batch_ocr_and_translate(images)
        # for i, text in enumerate(translated_texts):
        #     print(f"Panel {i+1}: {text}")
        
        print("Structure:")
        print("  translator = GeminiTranslator(api_key=api_key)")
        print("  images = [Image.open('panel1.png'), Image.open('panel2.png')]")
        print("  results = translator.batch_ocr_and_translate(images)")
        print("✓ Ready to process batches of images")
    except Exception as e:
        print(f"Error: {e}")


def example_with_manga_translator():
    """Example: Using Gemini through MangaTranslator."""
    print("\nExample 3: Using Gemini with MangaTranslator")
    print("=" * 60)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠ GEMINI_API_KEY environment variable not set")
        return
    
    try:
        from translator import MangaTranslator
        
        # Initialize with Gemini API key
        translator = MangaTranslator(gemini_api_key=api_key)
        
        # Use Gemini for translation
        # image = Image.open('manga_panel.png')
        # translated_text = translator.translate("", method="gemini", image=image)
        
        print("Structure:")
        print("  translator = MangaTranslator(gemini_api_key=api_key)")
        print("  image = Image.open('manga_panel.png')")
        print("  result = translator.translate('', method='gemini', image=image)")
        print("✓ Ready to use Gemini through MangaTranslator")
    except ImportError as e:
        print(f"Note: Some dependencies not installed: {e}")
        print("This is expected in minimal test environment")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Gemini API Integration Examples")
    print("=" * 60 + "\n")
    
    example_single_image()
    example_batch_processing()
    example_with_manga_translator()
    
    print("\n" + "=" * 60)
    print("Key Features:")
    print("  - OCR + Translation in single API call")
    print("  - Uses Gemini 2.0 Flash for fast responses")
    print("  - Supports batch processing of multiple images")
    print("  - Optimized with thinkingBudget=0 for speed")
    print("=" * 60)


if __name__ == "__main__":
    main()
