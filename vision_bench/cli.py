import argparse
from .models.mock_vlm import MockVLM
from .tasks.chartqa import ChartQATask
from .tasks.docvqa import DocVQATask
from .tasks.spatial import SpatialGroundingTask

def main():
    parser = argparse.ArgumentParser(
        prog="vlm-eval",
        description="multimodal-vision-bench: Standardized visual benchmark suite."
    )
    parser.add_argument("--model", type=str, default="mock-vlm", help="VLM identifier")
    parser.add_argument("--tasks", nargs="+", default=["chartqa", "docvqa", "spatial"], help="Tasks to run")

    args = parser.parse_args()

    vlm = MockVLM(args.model)
    task_map = {
        "chartqa": ChartQATask(),
        "docvqa": DocVQATask(),
        "spatial": SpatialGroundingTask()
    }

    print(f"\n[multimodal-vision-bench] Running VLM benchmark for '{args.model}'...")
    print("=" * 56)
    print("           MULTIMODAL BENCHMARK RESULTS")
    print("=" * 56)

    for task_name in args.tasks:
        if task_name in task_map:
            res = task_map[task_name].run_evaluation(vlm)
            print(f" Task: {res['task'].upper()} (Samples: {res['total']})")
            for k, v in res.items():
                if k not in ("task", "total", "details"):
                    val_str = f"{v * 100:.1f}%" if "acc" in k or "rate" in k or "sim" in k else str(v)
                    print(f"   - {k.replace('_', ' ').capitalize()}: {val_str}")
            print("-" * 56)

if __name__ == "__main__":
    main()
