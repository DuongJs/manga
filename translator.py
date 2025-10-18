from deep_translator import GoogleTranslator
from transformers import pipeline
import translators as ts
import random
import time
from gemini_translator import GeminiTranslator


class MangaTranslator:
    def __init__(self, gemini_api_key=None):
        self.target = "vi"
        self.source = "ja"
        self.gemini_api_key = gemini_api_key
        self.gemini_translator = None
        self.custom_prompt = None
        self.translators = {
            "google": self._translate_with_google,
            "hf": self._translate_with_hf,
            "sogou": self._translate_with_sogou,
            "bing": self._translate_with_bing,
            "gemini": self._translate_with_gemini
        }

    def set_custom_prompt(self, custom_prompt):
        """
        Set custom prompt for Gemini translation.
        
        Args:
            custom_prompt (str): Custom prompt for translation style/context
        """
        self.custom_prompt = custom_prompt
        if self.gemini_translator:
            self.gemini_translator.set_custom_prompt(custom_prompt)

    def translate(self, text, method="google", image=None):
        """
        Translates the given text to the target language using the specified method.

        Args:
            text (str): The text to be translated.
            method (str):"google" for Google Translator, 
                         "hf" for Helsinki-NLP's opus-mt-ja-en model (HF pipeline)
                         "sogou" for Sogou Translate
                         "bing" for Microsoft Bing Translator
                         "gemini" for Google Gemini (requires image for OCR+translation)
            image: PIL Image (required for gemini method)

        Returns:
            str: The translated text.
        """
        translator_func = self.translators.get(method)

        if translator_func:
            if method == "gemini":
                return translator_func(image)
            else:
                return translator_func(self._preprocess_text(text))
        else:
            raise ValueError("Invalid translation method.")
            
    def _translate_with_google(self, text):
        self._delay()
        translator = GoogleTranslator(source=self.source, target=self.target)
        translated_text = translator.translate(text)
        return translated_text if translated_text is not None else text

    def _translate_with_hf(self, text):
        pipe = pipeline("translation", model=f"Helsinki-NLP/opus-mt-ja-en")
        translated_text = pipe(text)[0]["translation_text"]
        return translated_text if translated_text is not None else text

    def _translate_with_sogou(self, text):
        self._delay()
        translated_text = ts.translate_text(text, translator="sogou",
                                            from_language=self.source,
                                            to_language=self.target)
        return translated_text if translated_text is not None else text

    def _translate_with_bing(self, text):
        self._delay()
        translated_text = ts.translate_text(text, translator="bing",
                                            from_language=self.source, 
                                            to_language=self.target)
        return translated_text if translated_text is not None else text

    def _translate_with_gemini(self, image):
        """
        Translate using Gemini API with OCR and translation in one call.
        
        Args:
            image: PIL Image containing the text
            
        Returns:
            str: Translated text
        """
        if image is None:
            return ""
        
        # Initialize Gemini translator if not already done
        if self.gemini_translator is None:
            try:
                self.gemini_translator = GeminiTranslator(api_key=self.gemini_api_key)
                if self.custom_prompt:
                    self.gemini_translator.set_custom_prompt(self.custom_prompt)
            except ValueError as e:
                print(f"Error initializing Gemini translator: {e}")
                return ""
        
        return self.gemini_translator.ocr_and_translate(image, target_lang=self.target, custom_prompt=self.custom_prompt)

    def _preprocess_text(self, text):
        preprocessed_text = text.replace("．", ".")
        return preprocessed_text

    def _delay(self):
        time.sleep(random.randint(3, 5))
