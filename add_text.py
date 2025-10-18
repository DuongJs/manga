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

    # Constants for text sizing
    TARGET_FILL_RATIO = 0.40  # Text should fill 40% of bubble height (reduced from 0.65)
    MIN_FONT_SIZE = 10  # Minimum readable font size (reduced from 12)
    MAX_FONT_SIZE = 40  # Maximum font size to prevent overly large text (reduced from 60)
    MIN_FONT_SIZE_THRESHOLD = 8  # Absolute minimum before giving up adjustment
    LINE_HEIGHT_MULTIPLIER = 1.1  # Line height as proportion of font size (reduced from 1.2 for tighter spacing)
    CHARS_PER_WIDTH_UNIT = 8  # Approximate characters per 10 pixels of width (decreased for fewer lines)
    
    target_height = h * TARGET_FILL_RATIO
    
    # Start with larger initial font size based on bubble height
    # Estimate number of lines based on text length and bubble width
    estimated_lines = max(1, len(text) // max(1, int(w / CHARS_PER_WIDTH_UNIT)))
    font_size = int(target_height / (estimated_lines * LINE_HEIGHT_MULTIPLIER))
    font_size = max(MIN_FONT_SIZE, min(font_size, MAX_FONT_SIZE))
    
    line_height = int(font_size * LINE_HEIGHT_MULTIPLIER)
    wrapping_ratio = 0.055  # Decreased from 0.075 to wrap later (fewer, longer lines)

    wrapped_text = textwrap.fill(text, width=int(w * wrapping_ratio), 
                                 break_long_words=True)
    
    font = ImageFont.truetype(font_path, size=font_size)

    lines = wrapped_text.split('\n')
    total_text_height = (len(lines)) * line_height

    # Adjust down if text is too large
    while total_text_height > h and font_size > MIN_FONT_SIZE_THRESHOLD:
        font_size = max(MIN_FONT_SIZE_THRESHOLD, font_size - 2)
        line_height = int(font_size * LINE_HEIGHT_MULTIPLIER)
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
