from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import cv2


def add_text(image, text, font_path, bubble_contour):
    """
    Add text inside a speech bubble contour.

    Args:
        image (numpy.ndarray): Processed bubble image (cv2 format - BGR).
        text (str): Text to be placed inside the speech bubble.
        font_path (str): Font path.
        bubble_contour (numpy.ndarray): Contour of the detected speech bubble.

    Returns:
        numpy.ndarray: Image with text placed inside the speech bubble.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    x, y, w, h = cv2.boundingRect(bubble_contour)

    # Target: text should fill approximately 80% of bubble height
    target_fill_ratio = 0.8
    target_height = h * target_fill_ratio
    
    # Start with larger initial font size based on bubble height
    # Estimate: each line is roughly 1.2x font_size in height
    estimated_lines = max(1, len(text) // max(1, int(w / 10)))  # Rough estimate
    font_size = int(target_height / (estimated_lines * 1.2))
    font_size = max(12, min(font_size, 60))  # Keep within reasonable bounds
    
    line_height = int(font_size * 1.2)
    wrapping_ratio = 0.075

    wrapped_text = textwrap.fill(text, width=int(w * wrapping_ratio), 
                                 break_long_words=True)
    
    font = ImageFont.truetype(font_path, size=font_size)

    lines = wrapped_text.split('\n')
    total_text_height = (len(lines)) * line_height

    # Adjust down if text is too large
    while total_text_height > h and font_size > 8:
        line_height = max(8, line_height - 2)
        font_size = max(8, font_size - 2)
        wrapping_ratio += 0.025

        wrapped_text = textwrap.fill(text, width=int(w * wrapping_ratio), 
                                 break_long_words=True)
                                 
        font = ImageFont.truetype(font_path, size=font_size)

        lines = wrapped_text.split('\n')
        total_text_height = (len(lines)) * line_height

    # Vertical centering
    text_y = y + (h - total_text_height) // 2

    for line in lines:
        text_length = draw.textlength(line, font=font)

        # Horizontal centering
        text_x = x + (w - text_length) // 2

        draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0))

        text_y += line_height

    image[:, :, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image
