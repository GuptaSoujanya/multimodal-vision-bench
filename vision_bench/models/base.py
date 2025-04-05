from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class VLMResponse:
    text: str
    predicted_boxes: Optional[List[List[float]]] = None
    metadata: Dict[str, Any] = None

class BaseVLM(ABC):
    """Abstract interface for Vision-Language Model inference."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def evaluate_sample(self, image_path: Optional[str], prompt: str) -> VLMResponse:
        pass
