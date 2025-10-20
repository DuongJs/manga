from ultralytics import YOLO
import numpy as np

# Global model cache
_model_cache = {}

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (list): [x1, y1, x2, y2, score, class_id]
        box2 (list): [x1, y1, x2, y2, score, class_id]
        
    Returns:
        float: IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def remove_duplicate_detections(detections, iou_threshold=0.3):
    """
    Remove duplicate detections using IoU-based filtering.
    Keeps detections with higher confidence scores.
    
    Args:
        detections (list): List of detections [x1, y1, x2, y2, score, class_id]
        iou_threshold (float): IoU threshold for considering boxes as duplicates
        
    Returns:
        list: Filtered list of detections with duplicates removed
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence score in descending order
    sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)
    
    keep = []
    suppress = [False] * len(sorted_detections)
    
    for i, current in enumerate(sorted_detections):
        if suppress[i]:
            continue
        
        keep.append(current)
        
        # Mark overlapping detections for suppression
        for j in range(i + 1, len(sorted_detections)):
            if not suppress[j]:
                iou = calculate_iou(current, sorted_detections[j])
                if iou >= iou_threshold:
                    suppress[j] = True
    
    return keep


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
    # YOLO performs NMS internally, but we add an additional layer of filtering
    bubbles = model(image_path, conf=conf_threshold, iou=iou_threshold)[0]

    # Get detections
    detections = bubbles.boxes.data.tolist()
    
    # Apply additional duplicate removal as a safety measure
    # This catches edge cases where YOLO's NMS might miss duplicates
    detections = remove_duplicate_detections(detections, iou_threshold)
    
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
    
    # Extract detections for each image and apply additional duplicate removal
    batch_detections = []
    for result in results:
        detections = result.boxes.data.tolist()
        # Apply additional duplicate removal as a safety measure
        detections = remove_duplicate_detections(detections, iou_threshold)
        batch_detections.append(detections)
    
    return batch_detections
