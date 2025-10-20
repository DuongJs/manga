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
import zipfile
import io
import tempfile
import shutil


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


# Batch processing functions
def create_zip_file(images, filenames):
    """
    Create a ZIP file containing all translated images.
    
    Args:
        images (list): List of PIL Images
        filenames (list): List of original filenames
        
    Returns:
        str: Path to the created ZIP file
    """
    # Create a temporary directory for storing images
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "translated_manga.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for idx, (img, filename) in enumerate(zip(images, filenames)):
            if img is not None:
                # Create a unique filename
                base_name = os.path.splitext(filename)[0] if filename else f"image_{idx+1}"
                img_filename = f"{base_name}_translated.png"
                
                # Save image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                # Write to zip
                zipf.writestr(img_filename, img_byte_arr.getvalue())
    
    return zip_path


def format_batch_stats(total, success, failed, processing_time):
    """
    Format batch processing statistics.
    
    Args:
        total (int): Total images processed
        success (int): Number of successful translations
        failed (int): Number of failed translations
        processing_time (float): Total processing time in seconds
        
    Returns:
        str: Formatted statistics string
    """
    stats = f"""
## 📊 Batch Processing Statistics

- **Total Images:** {total}
- **✅ Successful:** {success}
- **❌ Failed:** {failed}
- **Success Rate:** {(success/total*100):.1f}%
- **⏱️ Processing Time:** {processing_time:.2f}s
- **Average Time per Image:** {(processing_time/total):.2f}s
"""
    return stats


def predict_batch(images, translation_method, font, ocr_method, gemini_api_key=None, custom_prompt=None, conf_threshold=0.5, iou_threshold=0.3, progress=gr.Progress()):
    """
    Process multiple images in batch.
    
    Args:
        images (list): List of PIL Images uploaded via Gallery
        translation_method (str): Translation method to use
        font (str): Font path for text rendering
        ocr_method (str): OCR method to use
        gemini_api_key (str): Optional Gemini API key
        custom_prompt (str): Optional custom prompt for Gemini
        conf_threshold (float): Detection confidence threshold
        iou_threshold (float): IoU threshold for NMS
        progress: Gradio progress tracker
        
    Returns:
        tuple: (list of translated images, statistics markdown, zip file path)
    """
    import time
    start_time = time.time()
    
    if not images or len(images) == 0:
        return [], "❌ No images provided", None
    
    # Set defaults
    if translation_method is None:
        translation_method = "google"
    if font is None:
        font = "fonts/animeace_i.ttf"
    if ocr_method is None:
        ocr_method = "manga-ocr"
    
    # Validate Gemini API key if needed
    if translation_method == "gemini":
        if gemini_api_key:
            gemini_api_key = gemini_api_key.strip()
        
        if not gemini_api_key:
            print("No API key provided, trying API key manager...")
            gemini_api_key = api_key_manager.get_next_key()
            if gemini_api_key:
                print(f"✓ Retrieved API key from manager")
            else:
                gemini_api_key = os.getenv('GEMINI_API_KEY')
                if gemini_api_key:
                    print("✓ Retrieved API key from environment variable")
        
        if not gemini_api_key:
            return [], "❌ Gemini API key is required for Gemini translation", None
    
    # Initialize translator
    try:
        manga_translator = MangaTranslator(gemini_api_key=gemini_api_key)
        if custom_prompt and translation_method == "gemini":
            manga_translator.set_custom_prompt(custom_prompt)
    except ValueError as e:
        return [], f"❌ Translation setup error: {str(e)}", None
    
    # Initialize OCR
    if ocr_method == "paddleocr":
        ocr = get_paddle_ocr(lang='japan', use_textline_orientation=True, device='cpu')
    else:
        ocr = get_manga_ocr()
    
    # Load all images first
    loaded_images = []
    filenames = []
    
    progress(0.1, desc="Loading images...")
    
    for idx, img_path in enumerate(images):
        try:
            # Handle file path from gr.File component
            if isinstance(img_path, str):
                img = Image.open(img_path)
                filename = os.path.basename(img_path)
            elif hasattr(img_path, 'name'):  # File-like object
                img = Image.open(img_path.name)
                filename = os.path.basename(img_path.name)
            else:
                img = img_path if isinstance(img_path, Image.Image) else Image.open(img_path)
                filename = f'image_{idx+1}.png'
            
            loaded_images.append(img)
            filenames.append(filename)
        except Exception as e:
            print(f"Error loading image {idx+1}: {str(e)}")
            loaded_images.append(None)
            filenames.append(f'error_{idx+1}.png')
    
    total_images = len(loaded_images)
    
    # Use optimized batch processing for Gemini with bubble detection
    if translation_method == "gemini":
        progress(0.2, desc="Processing with Gemini batch API (optimized)...")
        translated_images = process_batch_with_gemini_optimized(
            loaded_images, filenames, manga_translator, font, 
            conf_threshold, iou_threshold, progress
        )
    else:
        # Process images one by one for other translation methods
        translated_images = []
        
        for idx, (img, filename) in enumerate(zip(loaded_images, filenames)):
            progress((0.2 + (idx / total_images) * 0.8), desc=f"Processing image {idx+1}/{total_images}")
            
            if img is None:
                translated_images.append(None)
                continue
            
            try:
                # Process single image
                result_image = process_single_image(
                    img, translation_method, font, ocr_method, 
                    manga_translator, ocr, 
                    conf_threshold, iou_threshold
                )
                
                translated_images.append(result_image)
                
            except Exception as e:
                print(f"Error processing image {idx+1}: {str(e)}")
                translated_images.append(None)
            
            # Clean up memory periodically
            if idx % 5 == 0:
                gc.collect()
    
    # Count successes and failures
    success_count = sum(1 for img in translated_images if img is not None)
    failed_count = total_images - success_count
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Generate statistics
    stats = format_batch_stats(total_images, success_count, failed_count, processing_time)
    
    # Create ZIP file if there are successful translations
    zip_path = None
    if success_count > 0:
        try:
            # Filter out None images
            valid_images = [(img, name) for img, name in zip(translated_images, filenames) if img is not None]
            if valid_images:
                valid_imgs, valid_names = zip(*valid_images)
                zip_path = create_zip_file(list(valid_imgs), list(valid_names))
        except Exception as e:
            print(f"Error creating ZIP file: {str(e)}")
            stats += f"\n\n⚠️ Warning: Could not create ZIP file: {str(e)}"
    
    # Filter out None images for gallery display
    display_images = [img for img in translated_images if img is not None]
    
    progress(1.0, desc="✅ Batch processing completed!")
    
    return display_images, stats, zip_path


def process_batch_with_gemini_optimized(images, filenames, manga_translator, font, conf_threshold, iou_threshold, progress):
    """
    Optimized batch processing using Gemini's batch API for all bubbles.
    
    Args:
        images: List of PIL Images
        filenames: List of filenames
        manga_translator: MangaTranslator instance
        font: Font path
        conf_threshold: Detection confidence threshold
        iou_threshold: IoU threshold
        progress: Progress callback
        
    Returns:
        list: List of translated PIL Images
    """
    translated_images = []
    total_images = len(images)
    
    # Step 1: Detect all bubbles in all images
    progress(0.3, desc="Detecting speech bubbles in all images...")
    
    all_bubbles_data = []  # Store (image_idx, bubble_coords, bubble_image)
    
    for img_idx, img in enumerate(images):
        if img is None:
            continue
            
        try:
            results = detect_bubbles(MODEL, img, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            image_array = np.array(img)
            
            for result in results:
                x1, y1, x2, y2, score, class_id = result
                detected_image = image_array[int(y1):int(y2), int(x1):int(x2)]
                bubble_img = Image.fromarray(detected_image.astype(np.uint8))
                
                all_bubbles_data.append({
                    'image_idx': img_idx,
                    'coords': (int(x1), int(y1), int(x2), int(y2)),
                    'bubble_image': bubble_img,
                    'detected_array': detected_image
                })
        except Exception as e:
            print(f"Error detecting bubbles in image {img_idx+1}: {str(e)}")
    
    # Step 2: Batch translate all bubbles using Gemini
    progress(0.5, desc=f"Translating {len(all_bubbles_data)} bubbles with Gemini batch API...")
    
    bubble_images = [b['bubble_image'] for b in all_bubbles_data]
    translated_texts = []
    
    if bubble_images:
        try:
            # Use Gemini's batch API if available
            if hasattr(manga_translator.gemini_translator, 'batch_ocr_and_translate'):
                translated_texts = manga_translator.gemini_translator.batch_ocr_and_translate(bubble_images)
            else:
                # Fallback to individual translation
                for bubble_img in bubble_images:
                    text = manga_translator.translate("", method="gemini", image=bubble_img)
                    translated_texts.append(text)
        except Exception as e:
            print(f"Error in batch translation: {str(e)}, falling back to individual processing")
            # Fallback to individual translation
            for bubble_img in bubble_images:
                try:
                    text = manga_translator.translate("", method="gemini", image=bubble_img)
                    translated_texts.append(text)
                except Exception as inner_e:
                    print(f"Error translating bubble: {inner_e}")
                    translated_texts.append("")
    
    # Step 3: Apply translations back to images
    progress(0.8, desc="Rendering translated text...")
    
    # Initialize output images
    image_arrays = {}
    for img_idx, img in enumerate(images):
        if img is not None:
            image_arrays[img_idx] = np.array(img)
    
    # Apply each translation
    for bubble_data, translated_text in zip(all_bubbles_data, translated_texts):
        img_idx = bubble_data['image_idx']
        x1, y1, x2, y2 = bubble_data['coords']
        detected_array = bubble_data['detected_array']
        
        try:
            # Process bubble and add text
            processed_bubble, cont = process_bubble(detected_array)
            rendered_bubble = add_text(processed_bubble, translated_text, font, cont)
            
            # Place back into original image
            image_arrays[img_idx][y1:y2, x1:x2] = rendered_bubble
        except Exception as e:
            print(f"Error rendering text for bubble: {str(e)}")
    
    # Convert back to PIL Images
    for img_idx, img in enumerate(images):
        if img is None:
            translated_images.append(None)
        elif img_idx in image_arrays:
            translated_images.append(Image.fromarray(image_arrays[img_idx]))
        else:
            translated_images.append(None)
    
    progress(0.95, desc="Finalizing...")
    gc.collect()
    
    return translated_images


def process_single_image(img, translation_method, font, ocr_method, manga_translator, ocr, conf_threshold, iou_threshold):
    """
    Process a single image (helper function for batch processing).
    
    Args:
        img: PIL Image
        translation_method: Translation method
        font: Font path
        ocr_method: OCR method
        manga_translator: MangaTranslator instance
        ocr: OCR instance
        conf_threshold: Detection confidence threshold
        iou_threshold: IoU threshold
        
    Returns:
        PIL Image: Translated image
    """
    # Detect bubbles
    results = detect_bubbles(MODEL, img, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    
    # Convert to numpy array
    image = np.array(img)
    
    # Process each detected bubble
    for result in results:
        x1, y1, x2, y2, score, class_id = result
        
        detected_image = image[int(y1):int(y2), int(x1):int(x2)]
        
        # Convert to PIL Image
        im = Image.fromarray(detected_image.astype(np.uint8))
        
        # Translate
        if translation_method == "gemini":
            text_translated = manga_translator.translate("", method=translation_method, image=im)
        else:
            text = ocr(im)
            text_translated = manga_translator.translate(text, method=translation_method)
        
        # Process bubble and add text
        detected_image, cont = process_bubble(detected_image)
        image[int(y1):int(y2), int(x1):int(x2)] = add_text(detected_image, text_translated, font, cont)
    
    # Convert back to PIL Image
    result_image = Image.fromarray(image)
    
    return result_image


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
    
    with gr.Tab("Batch Translate"):
        gr.Markdown("## 📚 Batch Translation - Translate Multiple Images at Once")
        gr.Markdown("Upload multiple manga images and translate them all with the same settings. Perfect for translating entire chapters!")
        
        with gr.Row():
            with gr.Column():
                # Image upload gallery
                batch_images = gr.File(
                    label="Upload Multiple Images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath"
                )
                
                gr.Markdown("### Translation Settings")
                
                batch_translation_method = gr.Dropdown(
                    TRANSLATION_METHODS,
                    label="Translation Method",
                    value="google",
                    info="Gemini recommended for batch processing"
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
                    placeholder="E.g., 'Use casual/informal Vietnamese style'",
                    lines=2
                )
                
                with gr.Accordion("Advanced Detection Settings", open=False):
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
                
                batch_translate_button = gr.Button("🚀 Start Batch Translation", variant="primary", size="lg")
            
            with gr.Column():
                # Statistics display
                batch_stats = gr.Markdown("📊 Upload images and click 'Start Batch Translation' to begin")
                
                # Preview gallery
                batch_output_gallery = gr.Gallery(
                    label="Translated Images Preview",
                    columns=3,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )
                
                # Download button
                batch_download_zip = gr.File(
                    label="📦 Download All Translated Images (ZIP)",
                    visible=True
                )
        
        # Batch translate button action
        batch_translate_button.click(
            fn=predict_batch,
            inputs=[
                batch_images,
                batch_translation_method,
                batch_font,
                batch_ocr_method,
                batch_api_key,
                batch_custom_prompt,
                batch_conf,
                batch_iou
            ],
            outputs=[batch_output_gallery, batch_stats, batch_download_zip]
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
