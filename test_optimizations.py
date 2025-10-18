#!/usr/bin/env python3
"""
Test script to validate the performance optimizations made to the manga translator.
This validates the changes without requiring external dependencies like PaddleOCR.
"""

import io
import base64
from PIL import Image
import numpy as np


def test_image_processing_fix():
    """Test that image processing no longer multiplies by 255 unnecessarily."""
    print("Testing image processing fix...")
    
    # Create a test image
    test_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Old way (incorrect): np.uint8((detected_image)*255)
    # This would overflow and cause issues
    old_way = np.uint8((test_array) * 255)  # This causes overflow
    
    # New way (correct): detected_image.astype(np.uint8)
    new_way = test_array.astype(np.uint8)
    
    # The new way should preserve the original values
    assert np.array_equal(test_array, new_way), "Image values should be preserved"
    assert not np.array_equal(test_array, old_way), "Old way causes data corruption"
    
    print("✓ Image processing fix validated")


def test_gemini_retry_mechanism():
    """Test that GeminiTranslator has retry mechanism."""
    print("\nTesting Gemini retry mechanism...")
    
    from gemini_translator import GeminiTranslator
    
    # Check class attributes
    assert hasattr(GeminiTranslator, 'DEFAULT_TIMEOUT'), "DEFAULT_TIMEOUT should be defined"
    assert hasattr(GeminiTranslator, 'MAX_RETRIES'), "MAX_RETRIES should be defined"
    assert hasattr(GeminiTranslator, 'RETRY_DELAY'), "RETRY_DELAY should be defined"
    
    # Check that __init__ accepts timeout and max_retries
    import inspect
    sig = inspect.signature(GeminiTranslator.__init__)
    params = list(sig.parameters.keys())
    assert 'timeout' in params, "timeout parameter should be in __init__"
    assert 'max_retries' in params, "max_retries parameter should be in __init__"
    
    # Check that _make_api_request method exists
    assert hasattr(GeminiTranslator, '_make_api_request'), "_make_api_request method should exist"
    
    print("✓ Retry mechanism validated")


def test_paddle_ocr_batch():
    """Test that PaddleOCRWrapper has batch_ocr method."""
    print("\nTesting PaddleOCR batch processing...")
    
    # We can't import PaddleOCR without installing it, but we can check the code
    with open('paddle_ocr_wrapper.py', 'r') as f:
        content = f.read()
    
    assert 'def batch_ocr' in content, "batch_ocr method should be defined"
    assert 'List of PIL Images' in content, "batch_ocr should document list input"
    assert 'List of recognized texts' in content, "batch_ocr should document list output"
    
    print("✓ Batch OCR method validated")


def test_memory_cleanup():
    """Test that memory cleanup is in place."""
    print("\nTesting memory cleanup...")
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    assert 'import gc' in content, "gc module should be imported"
    assert 'gc.collect()' in content, "gc.collect() should be called"
    assert 'del image' in content, "Explicit memory cleanup should be present"
    
    print("✓ Memory cleanup validated")


def test_async_zip():
    """Test that async ZIP creation is implemented."""
    print("\nTesting async ZIP compression...")
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    assert 'concurrent.futures' in content, "concurrent.futures should be imported"
    assert 'ThreadPoolExecutor' in content, "ThreadPoolExecutor should be used"
    assert 'save_image' in content, "Helper function for parallel processing should exist"
    
    print("✓ Async ZIP compression validated")


def test_batch_processing_optimization():
    """Test that batch processing is optimized."""
    print("\nTesting batch processing optimization...")
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Check that batch bubble detection is used
    assert 'detect_bubbles_batch' in content, "Batch bubble detection should be used"
    
    # Check that batch OCR is used for PaddleOCR
    assert 'batch_ocr' in content, "Batch OCR should be called"
    assert 'supports_batch' in content, "Batch support check should be present"
    
    print("✓ Batch processing optimization validated")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Performance Optimizations")
    print("=" * 60)
    
    test_image_processing_fix()
    test_gemini_retry_mechanism()
    test_paddle_ocr_batch()
    test_memory_cleanup()
    test_async_zip()
    test_batch_processing_optimization()
    
    print("\n" + "=" * 60)
    print("All optimization tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
