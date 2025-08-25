# multimodal-vision-bench

[![CI](https://github.com/GuptaSoujanya/multimodal-vision-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/GuptaSoujanya/multimodal-vision-bench/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A standardized benchmarking framework for Vision-Language Models (VLMs) targeting OCR precision, document layout comprehension (DocVQA), chart reasoning (ChartQA), and 2D spatial grounding.

---

## Evaluation Taxonomy

```mermaid
graph TD
    subgraph Tasks["Vision-Language Evaluation Tasks"]
        T1["ChartQA<br/>Numerical Reasoning & Plot Trends"]
        T2["DocVQA<br/>Dense Document & Invoice OCR"]
        T3["Spatial Grounding<br/>2D Bounding Box Localization"]
    end

    subgraph Metrics["Evaluation Metrics Engine"]
        M1["Relaxed Accuracy<br/>(±5% Numerical Margin)"]
        M2["Normalized Levenshtein / CER<br/>Character Error Rate"]
        M3["Bounding Box IoU<br/>Intersection over Union"]
    end

    T1 --> M1
    T2 --> M2
    T3 --> M3
```

---

## Benchmark Metrics

- **ChartQA Relaxed Accuracy:** Tolerates 5% numerical margin of error on plot and graph inquiries.
- **Document CER (Character Error Rate):** Normalized edit distance measuring OCR transcription degradation.
- **Spatial Bounding-Box IoU:** 2D polygon intersection over union for object localization.

---

## Installation

```bash
git clone https://github.com/GuptaSoujanya/multimodal-vision-bench.git
cd multimodal-vision-bench
pip install -e .
```

---

## Quickstart

```bash
python -m vision_bench.cli --tasks chartqa docvqa spatial
```

Sample Output:
```text
========================================================
           MULTIMODAL BENCHMARK RESULTS
========================================================
 Task: CHARTQA (Samples: 2)
   - Accuracy: 50.0%
--------------------------------------------------------
 Task: DOCVQA (Samples: 2)
   - Mean similarity: 74.5%
   - Mean cer: 0.255
--------------------------------------------------------
 Task: SPATIAL (Samples: 2)
   - Mean iou: 0.528
   - Iou success rate: 50.0%
--------------------------------------------------------
```

---

## Running Test Suite

```bash
python -m unittest discover tests -v
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
