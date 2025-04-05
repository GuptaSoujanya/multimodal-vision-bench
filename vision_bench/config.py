from dataclasses import dataclass
from typing import List

@dataclass
class VisionBenchConfig:
    model_name: str = "mock-vlm"
    tasks: List[str] = ("chartqa", "docvqa", "spatial")
    max_samples: int = 10
