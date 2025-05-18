from dataclasses import dataclass
from typing import List, Dict, Any
from ..metrics.relaxed_accuracy import relaxed_chart_match
from ..models.base import BaseVLM

@dataclass
class ChartSample:
    sample_id: str
    prompt: str
    target_answer: str

CHART_SAMPLES = [
    ChartSample("chart_01", "What is the peak revenue percentage depicted in the 2024 Q3 bar?", "42.0%"),
    ChartSample("chart_02", "What is the lowest value on the cost trend line?", "12.5")
]

class ChartQATask:
    name = "chartqa"

    def run_evaluation(self, vlm: BaseVLM) -> Dict[str, Any]:
        correct = 0
        details = []
        for s in CHART_SAMPLES:
            resp = vlm.evaluate_sample(image_path=None, prompt=s.prompt)
            match = relaxed_chart_match(resp.text, s.target_answer)
            if match:
                correct += 1
            details.append({"id": s.sample_id, "pred": resp.text, "target": s.target_answer, "correct": match})

        acc = correct / len(CHART_SAMPLES)
        return {"task": self.name, "accuracy": round(acc, 4), "total": len(CHART_SAMPLES), "details": details}
