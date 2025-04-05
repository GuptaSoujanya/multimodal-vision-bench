from typing import Optional, List
from .base import BaseVLM, VLMResponse

class MockVLM(BaseVLM):
    """Synthetic deterministic VLM provider for testing and validation."""

    def evaluate_sample(self, image_path: Optional[str], prompt: str) -> VLMResponse:
        p_lower = prompt.lower()
        if "chart" in p_lower or "percentage" in p_lower:
            return VLMResponse(text="42.5%")
        if "invoice" in p_lower or "total" in p_lower or "tax" in p_lower:
            return VLMResponse(text="$1,850.00")
        if "box" in p_lower or "detect" in p_lower or "localize" in p_lower:
            return VLMResponse(
                text="[0.12, 0.25, 0.45, 0.60]",
                predicted_boxes=[[0.12, 0.25, 0.45, 0.60]]
            )
        return VLMResponse(text="Recognized visual element.")
