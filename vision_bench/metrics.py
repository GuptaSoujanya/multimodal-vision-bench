import re

def calculate_cer(pred: str, target: str) -> float:
    if not target: return 0.0 if not pred else 1.0
    # Simple character mismatch ratio
    mismatches = sum(1 for a, b in zip(pred, target) if a != b) + abs(len(pred) - len(target))
    return round(min(1.0, mismatches / len(target)), 4)

def calculate_iou_2d(box1, box2):
    y1, x1, y2, x2 = box1
    y3, x3, y4, x4 = box2
    inter_h = max(0.0, min(y2, y4) - max(y1, y3))
    inter_w = max(0.0, min(x2, x4) - max(x1, x3))
    inter = inter_h * inter_w
    area1 = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    area2 = max(0.0, y4 - y3) * max(0.0, x4 - x3)
    union = area1 + area2 - inter
    return round(inter / union if union > 0 else 0.0, 4)

def relaxed_chart_accuracy(pred: str, target: str, tol=0.05):
    p = re.findall(r"[-+]?\d*\.?\d+", pred.replace(",", "").replace("%", ""))
    t = re.findall(r"[-+]?\d*\.?\d+", target.replace(",", "").replace("%", ""))
    if p and t:
        try:
            vp, vt = float(p[0]), float(t[0])
            return abs(vp - vt) / abs(vt) <= tol if vt != 0 else abs(vp - vt) < 1e-4
        except ValueError: pass
    return pred.strip().lower() == target.strip().lower()

# Custom CER threshold configuration helper
