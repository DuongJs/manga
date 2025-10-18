from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
import cv2


# Global font cache to avoid reloading fonts
_font_cache = {}


def get_cached_font(font_path, size):
    """
    Get a font from cache or load it if not cached.
    
    Args:
        font_path (str): Path to the font file
        size (int): Font size
        
    Returns:
        ImageFont: Cached or newly loaded font
    """
    cache_key = (font_path, size)
    if cache_key not in _font_cache:
        _font_cache[cache_key] = ImageFont.truetype(font_path, size=size)
    return _font_cache[cache_key]


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
    if bubble_contour is None:
        return image
    
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    x, y, w, h = cv2.boundingRect(bubble_contour)

    # Constants for text sizing
    TARGET_FILL_RATIO = 0.40  # Text should fill 40% of bubble height
    MIN_FONT_SIZE = 8  # Minimum readable font size
    MAX_FONT_SIZE = 40  # Maximum font size
    LINE_HEIGHT_MULTIPLIER = 1.1  # Line height as proportion of font size
    CHARS_PER_WIDTH_UNIT = 8  # Approximate characters per 10 pixels of width
    
    target_height = h * TARGET_FILL_RATIO
    
    # Estimate number of lines based on text length and bubble width
    estimated_lines = max(1, len(text) // max(1, int(w / CHARS_PER_WIDTH_UNIT)))
    
    # Binary search for optimal font size
    low, high = MIN_FONT_SIZE, MAX_FONT_SIZE
    best_font_size = MIN_FONT_SIZE
    best_wrapped_text = text
    best_lines = [text]
    
    while low <= high:
        mid = (low + high) // 2
        line_height = int(mid * LINE_HEIGHT_MULTIPLIER)
        wrapping_ratio = 0.055
        
        wrapped_text = textwrap.fill(text, width=int(w * wrapping_ratio), 
                                     break_long_words=True)
        lines = wrapped_text.split('\n')
        total_text_height = len(lines) * line_height
        
        # Check if this size fits
        if total_text_height <= h:
            # This size works, try larger
            best_font_size = mid
            best_wrapped_text = wrapped_text
            best_lines = lines
            low = mid + 1
        else:
            # Too large, try smaller
            high = mid - 1
    
    font_size = best_font_size
    line_height = int(font_size * LINE_HEIGHT_MULTIPLIER)
    wrapped_text = best_wrapped_text
    lines = best_lines
    
    # Get font from cache
    font = get_cached_font(font_path, font_size)
    
    total_text_height = len(lines) * line_height

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
