from ultralytics import YOLO
import numpy as np

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
    model = YOLO(model_path)
    # Run inference with adjusted confidence and IoU thresholds
    bubbles = model(image_path, conf=conf_threshold, iou=iou_threshold)[0]

    # Get detections
    detections = bubbles.boxes.data.tolist()
    
    # Additional filtering for overlapping boxes
    filtered_detections = []
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        
        # Check if this detection overlaps significantly with any already added
        is_duplicate = False
        for existing in filtered_detections:
            ex1, ey1, ex2, ey2, _, _ = existing
            
            # Calculate intersection over union (IoU)
            inter_x1 = max(x1, ex1)
            inter_y1 = max(y1, ey1)
            inter_x2 = min(x2, ex2)
            inter_y2 = min(y2, ey2)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                box1_area = (x2 - x1) * (y2 - y1)
                box2_area = (ex2 - ex1) * (ey2 - ey1)
                
                # Calculate IoU
                iou = inter_area / (box1_area + box2_area - inter_area)
                
                # If IoU is high, consider it a duplicate
                if iou > iou_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered_detections.append(det)
    
    return filtered_detections
