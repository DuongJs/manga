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
import zipfile
import io
from datetime import datetime


MODEL = "model.pt"
EXAMPLE_LIST = [["examples/0.png"],
                 ["examples/ex0.png"]]
TITLE = "Manga Translator"
DESCRIPTION = "Translate text in manga bubbles! Supports batch processing of multiple images. Vietnamese translation by default."

# Initialize API key manager
api_key_manager = APIKeyManager()

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
        if not gemini_api_key:
            # Try to get from API key manager
            gemini_api_key = api_key_manager.get_next_key()
            if not gemini_api_key:
                gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            return Image.fromarray(np.array(img))  # Return original image if no API key

    results = detect_bubbles(MODEL, img, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

    manga_translator = MangaTranslator(gemini_api_key=gemini_api_key)
    
    # Set custom prompt if provided
    if custom_prompt and translation_method == "gemini":
        manga_translator.set_custom_prompt(custom_prompt)
    
    # Initialize OCR based on selected method (not used for Gemini which has built-in OCR)
    if ocr_method == "paddleocr":
        ocr = PaddleOCRWrapper(lang='japan', use_textline_orientation=True, device='cpu')
    else:
        ocr = MangaOcr()

    image = np.array(img)

    for result in results:
        x1, y1, x2, y2, score, class_id = result

        detected_image = image[int(y1):int(y2), int(x1):int(x2)]

        im = Image.fromarray(np.uint8((detected_image)*255))
        
        # For Gemini, we skip separate OCR and pass the image directly
        if translation_method == "gemini":
            text_translated = manga_translator.translate("", method=translation_method, image=im)
        else:
            text = ocr(im)
            text_translated = manga_translator.translate(text, method=translation_method)

        detected_image, cont = process_bubble(detected_image)

        image[int(y1):int(y2), int(x1):int(x2)] = add_text(detected_image, text_translated, font, cont)

    return Image.fromarray(image)


def predict_batch(imgs, translation_method, font, ocr_method, gemini_api_key=None, custom_prompt=None, conf_threshold=0.5, iou_threshold=0.3):
    """
    Process multiple PIL images in batch.
    
    Args:
        imgs: List of PIL Image objects or single PIL Image
        translation_method: Translation method to use
        font: Font to use for text rendering
        ocr_method: OCR method to use
        gemini_api_key: Gemini API key (optional)
        custom_prompt: Custom prompt for Gemini translation
        conf_threshold: Confidence threshold for bubble detection
        iou_threshold: IoU threshold for duplicate removal
        
    Returns:
        List of processed PIL Image objects
    """
    if imgs is None:
        return []
    
    # Handle single image
    if not isinstance(imgs, list):
        imgs = [imgs]
    
    results = []
    for img in imgs:
        result = predict(img, translation_method, font, ocr_method, gemini_api_key, custom_prompt, conf_threshold, iou_threshold)
        results.append(result)
    
    return results


def predict_batch_files(file_paths, translation_method, font, ocr_method, gemini_api_key=None, custom_prompt=None, conf_threshold=0.5, iou_threshold=0.3):
    """
    Process multiple image files in batch with optimized Gemini batch processing.
    
    Args:
        file_paths: List of file paths or single file path
        translation_method: Translation method to use
        font: Font to use for text rendering
        ocr_method: OCR method to use
        gemini_api_key: Gemini API key (optional)
        custom_prompt: Custom prompt for Gemini translation
        conf_threshold: Confidence threshold for bubble detection
        iou_threshold: IoU threshold for duplicate removal
        
    Returns:
        List of processed images
    """
    if file_paths is None:
        return []
    
    # Handle single file path
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    
    # Load all images
    imgs = [Image.open(fp) for fp in file_paths]
    
    # If using Gemini, we can optimize batch processing
    if translation_method == "gemini":
        # Get API key
        if not gemini_api_key:
            gemini_api_key = api_key_manager.get_next_key()
            if not gemini_api_key:
                gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not gemini_api_key:
            # Fallback to normal processing
            return [predict(img, translation_method, font, ocr_method, gemini_api_key, custom_prompt, conf_threshold, iou_threshold) for img in imgs]
        
        # Process with batch optimization
        results = []
        from gemini_translator import GeminiTranslator
        gemini_translator = GeminiTranslator(api_key=gemini_api_key)
        if custom_prompt:
            gemini_translator.set_custom_prompt(custom_prompt)
        
        for img in imgs:
            # Detect bubbles
            bubble_results = detect_bubbles(MODEL, img, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            
            if not bubble_results:
                results.append(img)
                continue
            
            # Extract all bubble images
            image = np.array(img)
            bubble_images = []
            bubble_coords = []
            
            for result in bubble_results:
                x1, y1, x2, y2, score, class_id = result
                detected_image = image[int(y1):int(y2), int(x1):int(x2)]
                im = Image.fromarray(np.uint8((detected_image)*255))
                bubble_images.append(im)
                bubble_coords.append((int(y1), int(y2), int(x1), int(x2)))
            
            # Batch translate all bubbles in this image
            if bubble_images:
                translations = gemini_translator.batch_ocr_and_translate(bubble_images, custom_prompt=custom_prompt)
            else:
                translations = []
            
            # Apply translations back to image
            for i, result in enumerate(bubble_results):
                x1, y1, x2, y2, score, class_id = result
                detected_image = image[int(y1):int(y2), int(x1):int(x2)]
                text_translated = translations[i] if i < len(translations) else ""
                
                detected_image, cont = process_bubble(detected_image)
                image[int(y1):int(y2), int(x1):int(x2)] = add_text(detected_image, text_translated, font, cont)
            
            results.append(Image.fromarray(image))
        
        return results
    else:
        # Use normal processing for non-Gemini methods
        results = []
        for img in imgs:
            result = predict(img, translation_method, font, ocr_method, gemini_api_key, custom_prompt, conf_threshold, iou_threshold)
            results.append(result)
        return results


def download_all_images(images):
    """
    Create a ZIP file containing all processed images.
    
    Args:
        images: List of PIL Image objects
        
    Returns:
        Path to the ZIP file
    """
    if not images:
        return None
    
    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(images):
            # Save image to buffer
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Add to ZIP
            zip_file.writestr(f'translated_manga_{i+1}.png', img_buffer.getvalue())
    
    # Save ZIP to temporary file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_path = f'/tmp/manga_translated_{timestamp}.zip'
    
    with open(zip_path, 'wb') as f:
        f.write(zip_buffer.getvalue())
    
    return zip_path


# API Management functions
def add_api_key(api_key, name):
    """Add a new API key to the manager."""
    if api_key_manager.add_key(api_key, name):
        return "API key added successfully!", get_api_keys_display()
    return "Failed to add API key (may already exist)", get_api_keys_display()


def remove_api_key(index):
    """Remove an API key by index."""
    try:
        idx = int(index)
        if api_key_manager.remove_key(idx):
            return "API key removed successfully!", get_api_keys_display()
        return "Failed to remove API key (invalid index)", get_api_keys_display()
    except:
        return "Invalid index", get_api_keys_display()


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
    
    with gr.Tab("Single Image"):
        with gr.Row():
            with gr.Column():
                single_image = gr.Image(label="Upload Image", type="pil")
                single_translation_method = gr.Dropdown(
                    TRANSLATION_METHODS,
                    label="Translation Method",
                    value="google"
                )
                single_font = gr.Dropdown(
                    FONTS,
                    label="Text Font",
                    value="fonts/animeace_i.ttf"
                )
                single_ocr_method = gr.Dropdown(
                    OCR_METHODS,
                    label="OCR Method (not used for Gemini)",
                    value="manga-ocr"
                )
                single_api_key = gr.Textbox(
                    label="Gemini API Key (optional, uses API manager if not provided)",
                    type="password",
                    placeholder="Enter your Gemini API key for Gemini translation"
                )
                single_custom_prompt = gr.Textbox(
                    label="Custom Prompt for Gemini (optional)",
                    placeholder="E.g., 'Use casual/informal Vietnamese style' or 'Translate in a formal tone'",
                    lines=2
                )
                with gr.Row():
                    single_conf = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Detection Confidence Threshold"
                    )
                    single_iou = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.3,
                        step=0.05,
                        label="IoU Threshold (lower = less duplicates)"
                    )
                single_button = gr.Button("Translate")
            
            with gr.Column():
                single_output = gr.Image(label="Translated Image")
        
        gr.Examples(
            examples=EXAMPLE_LIST,
            inputs=[single_image]
        )
        
        single_button.click(
            fn=predict,
            inputs=[single_image, single_translation_method, single_font, single_ocr_method, single_api_key, single_custom_prompt, single_conf, single_iou],
            outputs=single_output
        )
    
    with gr.Tab("Batch Processing"):
        with gr.Row():
            with gr.Column():
                batch_images = gr.File(
                    label="Upload Images (multiple files)",
                    file_count="multiple",
                    type="filepath"
                )
                batch_translation_method = gr.Dropdown(
                    TRANSLATION_METHODS,
                    label="Translation Method",
                    value="google"
                )
                batch_font = gr.Dropdown(
                    FONTS,
                    label="Text Font",
                    value="fonts/animeace_i.ttf"
                )
                batch_ocr_method = gr.Dropdown(
                    OCR_METHODS,
                    label="OCR Method (not used for Gemini)",
                    value="manga-ocr"
                )
                batch_api_key = gr.Textbox(
                    label="Gemini API Key (optional, uses API manager if not provided)",
                    type="password",
                    placeholder="Enter your Gemini API key for Gemini translation"
                )
                batch_custom_prompt = gr.Textbox(
                    label="Custom Prompt for Gemini (optional)",
                    placeholder="E.g., 'Use casual/informal Vietnamese style' or 'Translate in a formal tone'",
                    lines=2
                )
                with gr.Row():
                    batch_conf = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Detection Confidence Threshold"
                    )
                    batch_iou = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.3,
                        step=0.05,
                        label="IoU Threshold (lower = less duplicates)"
                    )
                with gr.Row():
                    batch_button = gr.Button("Translate All", variant="primary")
                    download_button = gr.Button("Download All as ZIP")
            
            with gr.Column():
                batch_output = gr.Gallery(label="Translated Images", columns=2)
                download_file = gr.File(label="Download ZIP", visible=True)
        
        batch_button.click(
            fn=predict_batch_files,
            inputs=[batch_images, batch_translation_method, batch_font, batch_ocr_method, batch_api_key, batch_custom_prompt, batch_conf, batch_iou],
            outputs=batch_output
        )
        
        download_button.click(
            fn=download_all_images,
            inputs=[batch_output],
            outputs=download_file
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
