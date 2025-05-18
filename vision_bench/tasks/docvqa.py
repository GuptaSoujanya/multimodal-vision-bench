from dataclasses import dataclass
from typing import List, Dict, Any
from ..metrics.ocr_levenshtein import normalized_levenshtein_similarity, calculate_cer
from ..models.base import BaseVLM

@dataclass
class DocSample:
    sample_id: str
    prompt: str
    expected_text: str

DOC_SAMPLES = [
    DocSample("doc_01", "Extract invoice total amount due", "$1,850.00"),
    DocSample("doc_02", "What is the vendor tax registration number?", "TX-90210")
]

class DocVQATask:
    name = "docvqa"

    def run_evaluation(self, vlm: BaseVLM) -> Dict[str, Any]:
        similarities = []
        cers = []
        for s in DOC_SAMPLES:
            resp = vlm.evaluate_sample(image_path=None, prompt=s.prompt)
            sim = normalized_levenshtein_similarity(resp.text, s.expected_text)
            cer = calculate_cer(resp.text, s.expected_text)
            similarities.append(sim)
            cers.append(cer)

        mean_sim = sum(similarities) / len(similarities)
        mean_cer = sum(cers) / len(cers)
        return {
            "task": self.name,
            "mean_similarity": round(mean_sim, 4),
            "mean_cer": round(mean_cer, 4),
            "total": len(DOC_SAMPLES)
        }
