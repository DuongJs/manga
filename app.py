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
    Process multiple images in batch.
    
    Args:
        imgs: List of images or single image
        translation_method: Translation method to use
        font: Font to use for text rendering
        ocr_method: OCR method to use
        gemini_api_key: Gemini API key (optional)
        
    Returns:
        List of processed images
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

demo = gr.Interface(fn=predict,
                    inputs=["image",
                            gr.Dropdown([("Google", "google"),
                                         ("Helsinki-NLP's opus-mt-ja-en model",
                                          "hf"),
                                         ("Sogou", "sogou"),
                                         ("Bing", "bing"),
                                         ("Gemini (OCR + Translation)", "gemini")],
                                        label="Translation Method",
                                        value="google"),
                            gr.Dropdown([("animeace_i", "fonts/animeace_i.ttf"),
                                         ("mangati", "fonts/mangati.ttf"),
                                         ("ariali", "fonts/ariali.ttf")],
                                        label="Text Font",
                                        value="fonts/animeace_i.ttf"),
                            gr.Dropdown([("Manga OCR", "manga-ocr"),
                                         ("PaddleOCR", "paddleocr")],
                                        label="OCR Method (not used for Gemini)",
                                        value="manga-ocr"),
                            gr.Textbox(label="Gemini API Key (optional, or set GEMINI_API_KEY env var)",
                                      type="password",
                                      placeholder="Enter your Gemini API key for Gemini translation")
                            ],
                    outputs=[gr.Image()],
                    examples=EXAMPLE_LIST,
                    title=TITLE,
                    description=DESCRIPTION)


demo.launch(debug=False,
            share=False)
