import requests
import base64
import os
from PIL import Image
import io
import time
from functools import wraps


class GeminiTranslator:
    """
    Translator using Google's Gemini API for OCR and translation in a single API call.
    Supports multimodal input (images + text) and batch processing.
    """
    
    # Default timeout and retry configuration
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds (will use exponential backoff)
    BATCH_SIZE = 20  # Maximum number of images per API request for optimal performance
    
    def __init__(self, api_key=None, timeout=None, max_retries=None):
        """
        Initialize GeminiTranslator with API key.
        
        Args:
            api_key (str): Gemini API key. If None, will try to get from GEMINI_API_KEY env var.
            timeout (int): Request timeout in seconds. Default is 30.
            max_retries (int): Maximum number of retries for failed requests. Default is 3.
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Please:\n"
                "1. Add API keys via the 'API Key Management' tab in the web interface, OR\n"
                "2. Enter an API key in the 'Gemini API Key' field, OR\n"
                "3. Set the GEMINI_API_KEY environment variable\n"
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        # Validate API key format - basic validation
        if not isinstance(self.api_key, str) or len(self.api_key.strip()) == 0:
            raise ValueError(
                "Invalid API key format. API key must be a non-empty string.\n"
                "Get a valid API key from: https://makersuite.google.com/app/apikey"
            )
        
        self.api_key = self.api_key.strip()
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else self.MAX_RETRIES
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
        self.target_lang = "vi"
        self.source_lang = "ja"
        self.custom_prompt = None
    
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
    
    def _make_api_request(self, payload):
        """
        Make API request with retry logic and exponential backoff.
        
        Args:
            payload (dict): Request payload
            
        Returns:
            dict: API response
            
        Raises:
            ValueError: If request fails after all retries
        """
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url, 
                    json=payload, 
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                
                # Validate response structure
                if not isinstance(result, dict):
                    raise ValueError("Invalid API response format: response is not a dictionary")
                
                if 'candidates' not in result:
                    # Check for error in response
                    if 'error' in result:
                        error_info = result['error']
                        error_msg = error_info.get('message', str(error_info))
                        raise ValueError(f"Gemini API Error: {error_msg}")
                    raise ValueError("Invalid API response: missing 'candidates' field")
                
                if not result['candidates']:
                    raise ValueError("Invalid API response: empty candidates list")
                
                return result
                
            except requests.exceptions.Timeout:
                last_exception = ValueError(f"Request timeout after {self.timeout}s")
                if attempt < self.max_retries - 1:
                    wait_time = self.RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    print(f"Request timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                    
            except requests.exceptions.HTTPError as e:
                # Handle HTTP errors
                error_msg = f"Gemini API Error: {e}"
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        if 'error' in error_data:
                            error_info = error_data['error']
                            if 'message' in error_info:
                                error_msg = f"Gemini API Error: {error_info['message']}"
                                if 'API_KEY_INVALID' in str(error_info):
                                    error_msg = "Invalid Gemini API key. Please provide a valid API key from https://makersuite.google.com/app/apikey"
                                # Don't retry on authentication errors
                                raise ValueError(error_msg)
                    except ValueError:
                        raise
                    except Exception:
                        error_msg = f"Gemini API Error: {e.response.text}"
                
                last_exception = ValueError(error_msg)
                
                # Retry on server errors (5xx), but not on client errors (4xx)
                if e.response.status_code >= 500 and attempt < self.max_retries - 1:
                    wait_time = self.RETRY_DELAY * (2 ** attempt)
                    print(f"Server error, retrying in {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise last_exception
                    
            except requests.exceptions.RequestException as e:
                last_exception = ValueError(f"Network error calling Gemini API: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.RETRY_DELAY * (2 ** attempt)
                    print(f"Network error, retrying in {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
            except ValueError as e:
                # Don't retry on validation errors
                raise e
        
        # If we've exhausted all retries, raise the last exception
        if last_exception:
            print(f"All retry attempts failed")
            raise last_exception
        raise ValueError("Request failed with unknown error")
    
    def set_custom_prompt(self, custom_prompt):
        """
        Set a custom prompt for translation style/context.
        
        Args:
            custom_prompt (str): Custom prompt to guide translation style
        """
        self.custom_prompt = custom_prompt
    
    def ocr_and_translate(self, image, target_lang=None, custom_prompt=None):
        """
        Perform OCR and translation in a single API call using Gemini's multimodal capabilities.
        
        Args:
            image: PIL Image containing text to OCR and translate
            target_lang (str): Target language code (default: 'vi')
            custom_prompt (str): Optional custom prompt for translation style
            
        Returns:
            str: Translated text
        """
        target = target_lang or self.target_lang
        prompt_modifier = custom_prompt or self.custom_prompt or ""
        
        # Convert image to base64
        img_base64 = self._image_to_base64(image)
        
        # Create the prompt for OCR and translation
        system_instruction = (
            f"You are an expert manga translator. Extract all Japanese text from the manga image "
            f"and translate it to {target}. Only return the translated text without any explanations."
        )
        
        if prompt_modifier:
            system_instruction += f" {prompt_modifier}"
        
        prompt = (
            f"Extract all Japanese text from this manga panel and translate it to {target}. "
            f"Return only the translated text without any additional explanation or formatting."
        )
        
        if prompt_modifier:
            prompt += f" {prompt_modifier}"
        
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
        
        try:
            result = self._make_api_request(payload)
            
            # Extract the translated text from response
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        translated_text = parts[0]['text'].strip()
                        return translated_text
            
            return ""
            
        except ValueError as e:
            # Re-raise to propagate to UI
            raise e
    
    def batch_ocr_and_translate(self, images, target_lang=None, custom_prompt=None, batch_size=None):
        """
        Process multiple images in batched API calls for optimal efficiency.
        Automatically chunks images into batches of approximately 20 images per request.
        
        Args:
            images: List of PIL Images
            target_lang (str): Target language code (default: 'vi')
            custom_prompt (str): Optional custom prompt for translation style
            batch_size (int): Number of images per API request (default: 20)
            
        Returns:
            list: List of translated texts corresponding to each image
        """
        if not images:
            return []
        
        target = target_lang or self.target_lang
        prompt_modifier = custom_prompt or self.custom_prompt or ""
        chunk_size = batch_size or self.BATCH_SIZE
        
        total_images = len(images)
        
        # For very small batches (1-3 images), use individual processing to avoid parsing issues
        if total_images <= 3:
            print(f"Processing {total_images} images individually for better reliability")
            results = []
            for image in images:
                try:
                    translated = self.ocr_and_translate(image, target_lang, custom_prompt)
                    results.append(translated)
                except ValueError as e:
                    print(f"Error translating image: {e}")
                    results.append("")
            return results
        
        # Process images in chunks for optimal performance
        all_results = []
        num_chunks = (total_images + chunk_size - 1) // chunk_size  # Ceiling division
        
        print(f"Processing {total_images} images in {num_chunks} batch(es) of up to {chunk_size} images each")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_images)
            chunk_images = images[start_idx:end_idx]
            chunk_len = len(chunk_images)
            
            print(f"Processing batch {chunk_idx + 1}/{num_chunks}: images {start_idx + 1}-{end_idx} ({chunk_len} images)")
            
            # Process this chunk
            chunk_results = self._process_single_batch(chunk_images, target, prompt_modifier)
            all_results.extend(chunk_results)
        
        return all_results
    
    def _process_single_batch(self, images, target, prompt_modifier):
        """
        Process a single batch of images (internal method).
        
        Args:
            images: List of PIL Images (should be <= BATCH_SIZE)
            target: Target language
            prompt_modifier: Custom prompt modifier
            
        Returns:
            list: List of translated texts
        """
        if not images:
            return []
        
        # Convert all images to base64
        images_base64 = [self._image_to_base64(img) for img in images]
        
        # Use a more unique separator that's unlikely to appear in translations
        separator = "###TRANSLATION_SEPARATOR###"
        
        # Create the prompt for batch OCR and translation
        system_instruction = (
            f"You are an expert manga translator. For each manga panel image provided, "
            f"extract all Japanese text and translate it to {target}. "
            f"Return the translations separated by '{separator}' in the same order as the images. "
            f"Only return the translated text without any explanations."
        )
        
        if prompt_modifier:
            system_instruction += f" {prompt_modifier}"
        
        prompt = (
            f"Extract all Japanese text from these {len(images)} manga panels and translate each to {target}. "
            f"Return each translation separated by '{separator}' in the same order. "
            f"Do not include any additional explanation or formatting."
        )
        
        if prompt_modifier:
            prompt += f" {prompt_modifier}"
        
        # Build parts list with text prompt followed by all images
        parts = [{"text": prompt}]
        for img_base64 in images_base64:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_base64
                }
            })
        
        # Prepare the request payload
        payload = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [{"parts": parts}],
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingBudget": 0
                },
                "temperature": 0.3,
                "topP": 0.8,
                "topK": 10
            }
        }
        
        try:
            result = self._make_api_request(payload)
            
            # Extract the translated text from response
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if len(parts) > 0 and 'text' in parts[0]:
                        translated_text = parts[0]['text'].strip()
                        print(f"  Batch API response length: {len(translated_text)} chars")
                        
                        # Ensure proper UTF-8 encoding
                        if isinstance(translated_text, bytes):
                            translated_text = translated_text.decode('utf-8', errors='replace')
                        
                        # Split by separator
                        translations = [t.strip() for t in translated_text.split(separator)]
                        print(f"  Split into {len(translations)} translations for {len(images)} images")
                        
                        # Validate we got reasonable results
                        if len(translations) < len(images) * 0.5:  # Got less than half expected
                            print(f"  Warning: Only got {len(translations)} translations for {len(images)} images, falling back to individual processing")
                            raise ValueError("Insufficient translations from batch API")
                        
                        # Ensure we have the right number of translations
                        while len(translations) < len(images):
                            translations.append("")
                        
                        # Clean up translations - remove any control characters or invalid UTF-8
                        cleaned_translations = []
                        for trans in translations[:len(images)]:
                            # Remove control characters except newline and tab
                            cleaned = ''.join(char for char in trans if char.isprintable() or char in '\n\t')
                            cleaned_translations.append(cleaned)
                        
                        return cleaned_translations
            
            # Fallback to individual processing if batch fails
            print("  Batch processing failed (no valid response), falling back to individual processing")
            results = []
            for image in images:
                try:
                    translated = self.ocr_and_translate(image)
                    results.append(translated)
                except ValueError as e:
                    print(f"  Error translating image: {e}")
                    results.append("")
            return results
            
        except ValueError as e:
            # Fallback to individual processing on error
            print(f"  Batch processing error: {e}, falling back to individual processing")
            results = []
            for image in images:
                try:
                    translated = self.ocr_and_translate(image)
                    results.append(translated)
                except ValueError as inner_e:
                    print(f"  Error translating image: {inner_e}")
                    results.append("")
            return results
