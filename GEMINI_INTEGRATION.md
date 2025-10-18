# Gemini API Integration Documentation

## Overview

This manga translator now supports Google's Gemini API for advanced OCR and translation in a single API call. The integration uses Gemini's multimodal capabilities to process manga images directly.

## Features

1. **Combined OCR + Translation**: Performs OCR and translation in a single API call
2. **Batch Processing**: Process multiple images simultaneously
3. **Optimized Performance**: Uses `thinkingBudget: 0` for faster responses
4. **High Accuracy**: Leverages Gemini 2.0 Flash model

## Setup

### 1. Get API Key

Obtain your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 2. Set Environment Variable

```bash
export GEMINI_API_KEY='your-api-key-here'
```

Or pass the API key directly in the code/UI.

## Usage

### Method 1: Using GeminiTranslator Directly

```python
from gemini_translator import GeminiTranslator
from PIL import Image

# Initialize translator
translator = GeminiTranslator(api_key='your-api-key')

# Single image
image = Image.open('manga_panel.png')
translated_text = translator.ocr_and_translate(image)
print(translated_text)

# Batch processing
images = [Image.open(f'panel{i}.png') for i in range(1, 4)]
results = translator.batch_ocr_and_translate(images)
for i, text in enumerate(results):
    print(f"Panel {i+1}: {text}")
```

### Method 2: Using MangaTranslator

```python
from translator import MangaTranslator
from PIL import Image

# Initialize with Gemini API key
translator = MangaTranslator(gemini_api_key='your-api-key')

# Translate using Gemini (skips separate OCR step)
image = Image.open('manga_panel.png')
translated_text = translator.translate("", method="gemini", image=image)
print(translated_text)
```

### Method 3: Using the Web Interface

1. Run the application: `python app.py`
2. Open the web interface
3. Select "Gemini (OCR + Translation)" from the Translation Method dropdown
4. Enter your API key (if not set as environment variable)
5. Upload your manga image
6. Click Submit

## API Configuration

The Gemini translator uses the following configuration:

```json
{
  "model": "gemini-2.0-flash-exp",
  "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent",
  "generationConfig": {
    "thinkingConfig": {
      "thinkingBudget": 0
    },
    "temperature": 0.3,
    "topP": 0.8,
    "topK": 10
  }
}
```

### Configuration Parameters

- **model**: Uses Gemini 2.0 Flash experimental model for optimal speed/quality balance
- **thinkingBudget**: Set to 0 to disable extended thinking mode for faster responses
- **temperature**: 0.3 for consistent, deterministic translations
- **topP**: 0.8 for balanced creativity
- **topK**: 10 for controlled output diversity

## System Instructions

The translator uses optimized system instructions for manga translation:

```
You are an expert manga translator. Extract all Japanese text from the manga image 
and translate it to English. Only return the translated text without any explanations.
```

## How It Works

1. **Image Input**: Manga panel image is sent to Gemini API
2. **OCR**: Gemini extracts Japanese text using built-in OCR
3. **Translation**: Gemini translates extracted text to English
4. **Output**: Translated text is returned in a single response

This approach is more efficient than traditional OCR → Translation pipelines.

## Advantages Over Traditional Methods

1. **Single API Call**: No need for separate OCR and translation steps
2. **Context-Aware**: Gemini understands manga context for better translations
3. **Handles Complex Layouts**: Better at dealing with vertical text, sound effects, etc.
4. **No OCR Errors**: Eliminates errors from separate OCR preprocessing
5. **Multimodal Understanding**: Can understand visual context to improve translation

## Batch Processing

Process multiple images efficiently:

```python
translator = GeminiTranslator(api_key='your-api-key')

# Load multiple images
images = [
    Image.open('chapter1_page1.png'),
    Image.open('chapter1_page2.png'),
    Image.open('chapter1_page3.png'),
]

# Process all images
results = translator.batch_ocr_and_translate(images)

# Each result is the translated text from corresponding image
for i, translated_text in enumerate(results):
    print(f"Page {i+1}: {translated_text}")
```

## Error Handling

The implementation includes robust error handling:

- API key validation
- Request error handling
- Response parsing with fallbacks
- Graceful degradation if API is unavailable

## Cost Considerations

- Gemini API has usage limits and costs per API call
- Each image requires one API call
- Batch processing is sequential (not parallel) to respect rate limits
- Consider caching translations for repeated images

## Troubleshooting

### API Key Issues
- Ensure GEMINI_API_KEY is set correctly
- Verify API key is active and has quota

### Connection Errors
- Check internet connection
- Verify API endpoint is accessible
- Check for firewall/proxy restrictions

### Translation Quality
- Ensure images are clear and readable
- Higher resolution images generally work better
- Consider preprocessing images if quality is poor

## Examples

See `example_gemini_usage.py` for complete working examples.

## References

- [Gemini API Documentation](https://ai.google.dev/api/rest)
- [Gemini Multimodal Guide](https://ai.google.dev/gemini-api/docs/vision)
- [System Instructions Guide](https://ai.google.dev/gemini-api/docs/system-instructions)
