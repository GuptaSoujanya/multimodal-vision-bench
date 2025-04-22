from typing import List

def calculate_box_iou_2d(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) for 2D bounding boxes.
    Format: [ymin, xmin, ymax, xmax] normalized between 0.0 and 1.0.
    """
    y_min1, x_min1, y_max1, x_max1 = box1
    y_min2, x_min2, y_max2, x_max2 = box2

    inter_ymin = max(y_min1, y_min2)
    inter_xmin = max(x_min1, x_min2)
    inter_ymax = min(y_max1, y_max2)
    inter_xmax = min(x_max1, x_max2)

    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    intersection = inter_h * inter_w

    area1 = max(0.0, y_max1 - y_min1) * max(0.0, x_max1 - x_min1)
    area2 = max(0.0, y_max2 - y_min2) * max(0.0, x_max2 - x_min2)
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0
    return round(intersection / union, 4)
