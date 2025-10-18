from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2


# Global OCR cache to avoid reinitializing
_ocr_cache = {}


class PaddleOCRWrapper:
    """
    Wrapper class for PaddleOCR to provide OCR functionality for manga translation.
    This wrapper uses CPU-based processing following PaddleOCR best practices.
    """
    
    def __init__(self, lang='japan', use_textline_orientation=True, device='cpu'):
        """
        Initialize PaddleOCR with specified configuration.
        
        Args:
            lang (str): Language code for OCR. Default is 'japan' for Japanese manga.
            use_textline_orientation (bool): Whether to use textline orientation to handle rotated text.
            device (str): Device to use for processing. Default is 'cpu' for CPU-based processing.
        """
        # Use cache key to avoid reinitializing the same OCR configuration
        cache_key = f"{lang}_{use_textline_orientation}_{device}"
        if cache_key not in _ocr_cache:
            _ocr_cache[cache_key] = PaddleOCR(
                use_textline_orientation=use_textline_orientation,
                lang=lang,
                device=device,
                show_log=False
            )
        self.ocr = _ocr_cache[cache_key]
    
    def __call__(self, image):
        """
        Perform OCR on the given image.
        
        Args:
            image: PIL Image or numpy array containing the text to recognize.
            
        Returns:
            str: Recognized text from the image.
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in the correct format (BGR for PaddleOCR)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Perform OCR
        result = self.ocr.ocr(image, cls=True)
        
        # Extract text from results
        if result and result[0]:
            # Concatenate all detected text lines
            texts = [line[1][0] for line in result[0]]
            return ' '.join(texts)
        
        return ""
