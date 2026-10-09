import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.error_analysis import analyze_errors

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze SP-DocVQA prediction errors."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/evaluation/layoutlmv3/predictions_per_question.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation/layoutlmv3"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--anls-threshold",
        type=float,
        default=0.5
    )

    return parser.parse_args()

def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT / path

def load_predictions(predictions_path: Path) -> List[Dict]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")

    with predictions_path.open("r", encoding="utf-8") as file:
        predictions = json.load(file)

    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list.")

    return predictions

def main():
    args = parse_args()

    if not 0 <= args.iou_threshold <= 1:
        raise ValueError("IoU threshold must be between 0 and 1.")

    if not 0 <= args.anls_threshold <= 1:
        raise ValueError("ANLS threshold must be between 0 and 1.")

    predictions_path = resolve_path(args.predictions)
    output_dir = resolve_path(args.output_dir)
    predictions = load_predictions(predictions_path)

    if not predictions:
        raise ValueError("No predictions to analyze.")

    output_dir.mkdir(parents=True, exist_ok=True)

    categorized, summary = analyze_errors(
        predictions=predictions,
        iou_threshold=args.iou_threshold,
        anls_threshold=args.anls_threshold
    )

    cases_path = output_dir / "error_cases.json"
    summary_path = output_dir / "error_summary.json"

    with cases_path.open("w", encoding="utf-8") as file:
        json.dump(categorized, file, indent=2, ensure_ascii=False)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS RESULTS")
    print("=" * 60)
    print("Total samples:", summary["total_samples"])

    for category, count in summary["category_counts"].items():
        percentage = summary["category_percentages"][category]
        print(f"{category:30s}: {count:5d} ({percentage:.2f}%)")

    print("=" * 60)
    print("Cases:", cases_path)
    print("Summary:", summary_path)

if __name__ == "__main__":
    main()