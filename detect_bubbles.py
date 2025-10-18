from ultralytics import YOLO
import numpy as np

# Global model cache
_model_cache = {}

def detect_bubbles(model_path, image_path, conf_threshold=0.5, iou_threshold=0.3):
    """
    Detects bubbles in an image using a YOLOv8 model with improved duplicate filtering.
    Args:
        model_path (str): The file path to the YOLO model.
        image_path (str): The file path to the input image.
        conf_threshold (float): Confidence threshold for detections (default: 0.5)
        iou_threshold (float): IoU threshold for non-maximum suppression (default: 0.3)
    Returns:
        list: A list containing the coordinates, score, and class_id of 
              the detected bubbles with duplicates removed.
    """
    # Cache model globally to avoid reloading
    if model_path not in _model_cache:
        _model_cache[model_path] = YOLO(model_path)
    model = _model_cache[model_path]
    
    # Run inference with adjusted confidence and IoU thresholds
    # YOLO already performs NMS internally with the iou parameter, so we don't need additional filtering
    bubbles = model(image_path, conf=conf_threshold, iou=iou_threshold)[0]

    # Get detections - no additional NMS needed as YOLO handles it
    detections = bubbles.boxes.data.tolist()
    
    return detections


def detect_bubbles_batch(model_path, image_paths, conf_threshold=0.5, iou_threshold=0.3):
    """
    Detects bubbles in multiple images using a single YOLO batch inference call.
    
    Args:
        model_path (str): The file path to the YOLO model.
        image_paths (list): List of image paths or PIL images.
        conf_threshold (float): Confidence threshold for detections (default: 0.5)
        iou_threshold (float): IoU threshold for non-maximum suppression (default: 0.3)
    Returns:
        list: A list of detection lists, one for each input image.
    """
    # Cache model globally to avoid reloading
    if model_path not in _model_cache:
        _model_cache[model_path] = YOLO(model_path)
    model = _model_cache[model_path]
    
    # Run batch inference - YOLO can process multiple images at once
    results = model(image_paths, conf=conf_threshold, iou=iou_threshold)
    
    # Extract detections for each image
    batch_detections = []
    for result in results:
        detections = result.boxes.data.tolist()
        batch_detections.append(detections)
    
    return batch_detections
