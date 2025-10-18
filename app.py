from add_text import add_text
from detect_bubbles import detect_bubbles
from process_bubble import process_bubble
from translator import MangaTranslator
from ultralytics import YOLO
from manga_ocr import MangaOcr
from paddle_ocr_wrapper import PaddleOCRWrapper
from PIL import Image
from api_key_manager import APIKeyManager
import gradio as gr
import numpy as np
import cv2
import os
import gc
from datetime import datetime


MODEL = "model.pt"
EXAMPLE_LIST = [["examples/0.png"],
                 ["examples/ex0.png"]]
TITLE = "Manga Translator"
DESCRIPTION = "Translate text in manga bubbles! Vietnamese translation by default."

# Initialize API key manager
api_key_manager = APIKeyManager()

# Global model caches for reuse
_manga_ocr_cache = None
_paddle_ocr_cache = {}

def get_manga_ocr():
    """Get or initialize cached MangaOcr instance."""
    global _manga_ocr_cache
    if _manga_ocr_cache is None:
        _manga_ocr_cache = MangaOcr()
    return _manga_ocr_cache

def get_paddle_ocr(lang='japan', use_textline_orientation=True, device='cpu'):
    """Get or initialize cached PaddleOCR instance."""
    global _paddle_ocr_cache
    cache_key = f"{lang}_{use_textline_orientation}_{device}"
    if cache_key not in _paddle_ocr_cache:
        _paddle_ocr_cache[cache_key] = PaddleOCRWrapper(lang=lang, use_textline_orientation=use_textline_orientation, device=device)
    return _paddle_ocr_cache[cache_key]

# Dropdown options
TRANSLATION_METHODS = [
    ("Google", "google"),
    ("Helsinki-NLP's opus-mt-ja-en model", "hf"),
    ("Sogou", "sogou"),
    ("Bing", "bing"),
    ("Gemini (OCR + Translation)", "gemini")
]

FONTS = [
    ("animeace_i", "fonts/animeace_i.ttf"),
    ("mangati", "fonts/mangati.ttf"),
    ("ariali", "fonts/ariali.ttf")
]

OCR_METHODS = [
    ("Manga OCR", "manga-ocr"),
    ("PaddleOCR", "paddleocr")
]


def predict(img, translation_method, font, ocr_method, gemini_api_key=None, custom_prompt=None, conf_threshold=0.5, iou_threshold=0.3):
    if translation_method == None:
        translation_method = "google"
    if font == None:
        font = "fonts/animeace_i.ttf"
    if ocr_method == None:
        ocr_method = "manga-ocr"

    # Validate Gemini API key if Gemini is selected
    if translation_method == "gemini":
        # Strip whitespace from API key if provided
        if gemini_api_key:
            gemini_api_key = gemini_api_key.strip()
        
        if not gemini_api_key:
            # Try to get from API key manager
            print("No API key provided in form, trying API key manager...")
            gemini_api_key = api_key_manager.get_next_key()
            if gemini_api_key:
                print(f"✓ Retrieved API key from manager (key starts with: {gemini_api_key[:10]}...)")
            else:
                print("✗ No API key available in manager, trying environment variable...")
                gemini_api_key = os.getenv('GEMINI_API_KEY')
                if gemini_api_key:
                    print("✓ Retrieved API key from environment variable")
        else:
            print(f"Using API key provided in form (key starts with: {gemini_api_key[:10]}...)")
        
        if not gemini_api_key:
            raise gr.Error("Gemini API key is required. Please enter an API key or add one in the API Key Management tab.")

    results = detect_bubbles(MODEL, img, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

    try:
        manga_translator = MangaTranslator(gemini_api_key=gemini_api_key)
        
        # Set custom prompt if provided
        if custom_prompt and translation_method == "gemini":
            manga_translator.set_custom_prompt(custom_prompt)
    except ValueError as e:
        raise gr.Error(f"Translation setup error: {str(e)}")
    
    # Initialize OCR based on selected method (not used for Gemini which has built-in OCR)
    if ocr_method == "paddleocr":
        ocr = get_paddle_ocr(lang='japan', use_textline_orientation=True, device='cpu')
    else:
        ocr = get_manga_ocr()

    image = np.array(img)

    for result in results:
        x1, y1, x2, y2, score, class_id = result

        detected_image = image[int(y1):int(y2), int(x1):int(x2)]

        # Image is already in uint8 format [0-255], no need to multiply by 255
        im = Image.fromarray(detected_image.astype(np.uint8))
        
        try:
            # For Gemini, we skip separate OCR and pass the image directly
            if translation_method == "gemini":
                text_translated = manga_translator.translate("", method=translation_method, image=im)
            else:
                text = ocr(im)
                text_translated = manga_translator.translate(text, method=translation_method)
        except ValueError as e:
            raise gr.Error(f"Translation error: {str(e)}")

        detected_image, cont = process_bubble(detected_image)

        image[int(y1):int(y2), int(x1):int(x2)] = add_text(detected_image, text_translated, font, cont)

    result_image = Image.fromarray(image)
    
    # Clean up memory
    del image
    gc.collect()
    
    return result_image

# API Management functions
def add_api_key(api_key, name):
    """Add a new API key to the manager."""
    # Strip whitespace from API key and name
    if api_key:
        api_key = api_key.strip()
    if name:
        name = name.strip()
    
    if not api_key:
        return "API key cannot be empty", get_api_keys_display()
    
    try:
        if api_key_manager.add_key(api_key, name):
            return "API key added successfully!", get_api_keys_display()
        return "Failed to add API key (may already exist)", get_api_keys_display()
    except Exception as e:
        return f"Error saving API key: {str(e)}", get_api_keys_display()


def remove_api_key(index):
    """Remove an API key by index."""
    try:
        idx = int(index)
        if api_key_manager.remove_key(idx):
            return "API key removed successfully!", get_api_keys_display()
        return "Failed to remove API key (invalid index)", get_api_keys_display()
    except ValueError:
        return "Invalid index format", get_api_keys_display()
    except Exception as e:
        return f"Error removing API key: {str(e)}", get_api_keys_display()


def get_api_keys_display():
    """Get formatted display of all API keys."""
    keys = api_key_manager.get_all_keys()
    stats = api_key_manager.get_stats()
    
    if not keys:
        return "No API keys configured."
    
    display = f"**Total Keys:** {stats['total_keys']} | **Total Usage:** {stats['total_usage']} | **Current Index:** {stats['current_key_index']}\n\n"
    display += "| Index | Name | Key Preview | Usage Count | Added At |\n"
    display += "|-------|------|-------------|-------------|----------|\n"
    
    for key in keys:
        display += f"| {key['index']} | {key['name']} | {key['key']} | {key['usage_count']} | {key['added_at']} |\n"
    
    return display


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)
    
    with gr.Tab("Translate"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="pil")
                translation_method = gr.Dropdown(
                    TRANSLATION_METHODS,
                    label="Translation Method",
                    value="google"
                )
                font = gr.Dropdown(
                    FONTS,
                    label="Text Font",
                    value="fonts/animeace_i.ttf"
                )
                ocr_method = gr.Dropdown(
                    OCR_METHODS,
                    label="OCR Method (not used for Gemini)",
                    value="manga-ocr"
                )
                api_key = gr.Textbox(
                    label="Gemini API Key (optional, uses API manager if not provided)",
                    type="password",
                    placeholder="Enter your Gemini API key for Gemini translation"
                )
                custom_prompt = gr.Textbox(
                    label="Custom Prompt for Gemini (optional)",
                    placeholder="E.g., 'Use casual/informal Vietnamese style' or 'Translate in a formal tone'",
                    lines=2
                )
                with gr.Row():
                    conf = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Detection Confidence Threshold"
                    )
                    iou = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.3,
                        step=0.05,
                        label="IoU Threshold (lower = less duplicates)"
                    )
                translate_button = gr.Button("Translate", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Translated Image")
        
        gr.Examples(
            examples=EXAMPLE_LIST,
            inputs=[image_input]
        )
        
        translate_button.click(
            fn=predict,
            inputs=[image_input, translation_method, font, ocr_method, api_key, custom_prompt, conf, iou],
            outputs=output_image
        )
    
    with gr.Tab("API Key Management"):
        gr.Markdown("## Manage Gemini API Keys")
        gr.Markdown("Add multiple API keys for round-robin usage. Keys are saved to `api_keys.json`.")
        
        with gr.Row():
            with gr.Column():
                api_key_input = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="Enter Gemini API key"
                )
                api_key_name = gr.Textbox(
                    label="Key Name (optional)",
                    placeholder="E.g., 'Personal Key', 'Work Key'"
                )
                add_key_button = gr.Button("Add API Key", variant="primary")
                
                gr.Markdown("---")
                
                remove_index = gr.Number(
                    label="Key Index to Remove",
                    value=0,
                    precision=0
                )
                remove_key_button = gr.Button("Remove API Key", variant="stop")
            
            with gr.Column():
                api_status = gr.Textbox(label="Status", lines=2)
                api_keys_display = gr.Markdown(value=get_api_keys_display())
        
        add_key_button.click(
            fn=add_api_key,
            inputs=[api_key_input, api_key_name],
            outputs=[api_status, api_keys_display]
        )
        
        remove_key_button.click(
            fn=remove_api_key,
            inputs=[remove_index],
            outputs=[api_status, api_keys_display]
        )
        
        # Refresh button
        refresh_button = gr.Button("Refresh Display")
        refresh_button.click(
            fn=lambda: get_api_keys_display(),
            inputs=[],
            outputs=[api_keys_display]
        )


demo.launch(debug=False,
            share=False)
