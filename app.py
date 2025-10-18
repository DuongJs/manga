from add_text import add_text
from detect_bubbles import detect_bubbles
from process_bubble import process_bubble
from translator import MangaTranslator
from ultralytics import YOLO
from manga_ocr import MangaOcr
from paddle_ocr_wrapper import PaddleOCRWrapper
from PIL import Image
import gradio as gr
import numpy as np
import cv2
import os


MODEL = "model.pt"
EXAMPLE_LIST = [["examples/0.png"],
                 ["examples/ex0.png"]]
TITLE = "Manga Translator"
DESCRIPTION = "Translate text in manga bubbles! Supports batch processing of multiple images."

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


def predict(img, translation_method, font, ocr_method, gemini_api_key=None):
    if translation_method == None:
        translation_method = "google"
    if font == None:
        font = "fonts/animeace_i.ttf"
    if ocr_method == None:
        ocr_method = "manga-ocr"

    # Validate Gemini API key if Gemini is selected
    if translation_method == "gemini" and not gemini_api_key:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            return Image.fromarray(np.array(img))  # Return original image if no API key

    results = detect_bubbles(MODEL, img)

    manga_translator = MangaTranslator(gemini_api_key=gemini_api_key)
    
    # Initialize OCR based on selected method (not used for Gemini which has built-in OCR)
    if ocr_method == "paddleocr":
        ocr = PaddleOCRWrapper(lang='japan', use_angle_cls=True, use_gpu=False)
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


def predict_batch(imgs, translation_method, font, ocr_method, gemini_api_key=None):
    """
    Process multiple PIL images in batch.
    
    Args:
        imgs: List of PIL Image objects or single PIL Image
        translation_method: Translation method to use
        font: Font to use for text rendering
        ocr_method: OCR method to use
        gemini_api_key: Gemini API key (optional)
        
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
        result = predict(img, translation_method, font, ocr_method, gemini_api_key)
        results.append(result)
    
    return results


def predict_batch_files(file_paths, translation_method, font, ocr_method, gemini_api_key=None):
    """
    Process multiple image files in batch.
    
    Args:
        file_paths: List of file paths or single file path
        translation_method: Translation method to use
        font: Font to use for text rendering
        ocr_method: OCR method to use
        gemini_api_key: Gemini API key (optional)
        
    Returns:
        List of processed images
    """
    if file_paths is None:
        return []
    
    # Handle single file path
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    
    results = []
    for file_path in file_paths:
        # Load image from file path
        img = Image.open(file_path)
        result = predict(img, translation_method, font, ocr_method, gemini_api_key)
        results.append(result)
    
    return results


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
                    label="Gemini API Key (optional, or set GEMINI_API_KEY env var)",
                    type="password",
                    placeholder="Enter your Gemini API key for Gemini translation"
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
            inputs=[single_image, single_translation_method, single_font, single_ocr_method, single_api_key],
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
                    label="Gemini API Key (optional, or set GEMINI_API_KEY env var)",
                    type="password",
                    placeholder="Enter your Gemini API key for Gemini translation"
                )
                batch_button = gr.Button("Translate All")
            
            with gr.Column():
                batch_output = gr.Gallery(label="Translated Images", columns=2)
        
        batch_button.click(
            fn=predict_batch_files,
            inputs=[batch_images, batch_translation_method, batch_font, batch_ocr_method, batch_api_key],
            outputs=batch_output
        )


demo.launch(debug=False,
            share=False)
