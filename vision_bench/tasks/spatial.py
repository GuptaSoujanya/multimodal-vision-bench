from dataclasses import dataclass
from typing import List, Dict, Any
from ..metrics.box_iou import calculate_box_iou_2d
from ..models.base import BaseVLM

@dataclass
class SpatialSample:
    sample_id: str
    prompt: str
    ground_truth_box: List[float] # [ymin, xmin, ymax, xmax]

SPATIAL_SAMPLES = [
    SpatialSample("box_01", "Detect bounding box for navigation chart header", [0.10, 0.20, 0.40, 0.65]),
    SpatialSample("box_02", "Localize legend key bounding box", [0.80, 0.15, 0.95, 0.45])
]

class SpatialGroundingTask:
    name = "spatial"

    def run_evaluation(self, vlm: BaseVLM, iou_threshold: float = 0.5) -> Dict[str, Any]:
        ious = []
        hits = 0
        for s in SPATIAL_SAMPLES:
            resp = vlm.evaluate_sample(image_path=None, prompt=s.prompt)
            pred_box = resp.predicted_boxes[0] if resp.predicted_boxes else [0.0, 0.0, 0.0, 0.0]
            iou = calculate_box_iou_2d(pred_box, s.ground_truth_box)
            ious.append(iou)
            if iou >= iou_threshold:
                hits += 1

        return {
            "task": self.name,
            "mean_iou": round(sum(ious) / len(ious), 4),
            "iou_success_rate": round(hits / len(ious), 4),
            "total": len(SPATIAL_SAMPLES)
        }
