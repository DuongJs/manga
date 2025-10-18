import requests
import base64
import os
from PIL import Image
import io


class GeminiTranslator:
    """
    Translator using Google's Gemini API for OCR and translation in a single API call.
    Supports multimodal input (images + text) and batch processing.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize GeminiTranslator with API key.
        
        Args:
            api_key (str): Gemini API key. If None, will try to get from GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        self.target_lang = "en"
        self.source_lang = "ja"
    
    def _image_to_base64(self, image):
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Base64 encoded image string
        """
        if isinstance(image, Image.Image):
            buffered = io.BytesIO()
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str
        else:
            raise ValueError("Input must be a PIL Image")
    
    def ocr_and_translate(self, image, target_lang=None):
        """
        Perform OCR and translation in a single API call using Gemini's multimodal capabilities.
        
        Args:
            image: PIL Image containing text to OCR and translate
            target_lang (str): Target language code (default: 'en')
            
        Returns:
            str: Translated text
        """
        target = target_lang or self.target_lang
        
        # Convert image to base64
        img_base64 = self._image_to_base64(image)
        
        # Create the prompt for OCR and translation
        system_instruction = (
            f"You are an expert manga translator. Extract all Japanese text from the manga image "
            f"and translate it to {target}. Only return the translated text without any explanations."
        )
        
        prompt = (
            f"Extract all Japanese text from this manga panel and translate it to {target}. "
            f"Return only the translated text without any additional explanation or formatting."
        )
        
        # Prepare the request payload
        payload = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingBudget": 0
                },
                "temperature": 0.3,
                "topP": 0.8,
                "topK": 10
            }
        }
        
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the translated text from response
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        translated_text = parts[0]['text'].strip()
                        return translated_text
            
            return ""
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Gemini API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return ""
    
    def batch_ocr_and_translate(self, images, target_lang=None):
        """
        Process multiple images in batch.
        
        Args:
            images: List of PIL Images
            target_lang (str): Target language code (default: 'en')
            
        Returns:
            list: List of translated texts corresponding to each image
        """
        results = []
        for image in images:
            translated = self.ocr_and_translate(image, target_lang)
            results.append(translated)
        return results
