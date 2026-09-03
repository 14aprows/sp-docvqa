import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.visualize_prediction import visualize_prediction

ERROR_CATEGORIES = [
    "all",
    "correct",
    "text_correct_location_wrong",
    "text_wrong_location_correct",
    "partially_correct_text",
    "both_wrong"
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualisasi hasil prediksi dan analisis kesalahan SP-DocVQA."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/evaluation/layoutlmv3_spdocvqa/error_cases.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations/layoutlmv3_spdocvqa"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=ERROR_CATEGORIES
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20
    )

    return parser.parse_args()

def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT / path

def load_predictions(predictions_path: Path) -> List[Dict]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions tidak ditemukan: {predictions_path}")

    with predictions_path.open("r", encoding="utf-8") as file:
        predictions = json.load(file)

    if not isinstance(predictions, list):
        raise ValueError("Format predictions harus berupa list.")

    return predictions

def safe_filename(value) -> str:
    filename = re.sub(r"[^a-zA-Z0-9_-]", "_", str(value))
    return filename[:100]

def main():
    args = parse_args()

    if args.limit < 0:
        raise ValueError("Limit tidak boleh negatif.")

    predictions_path = resolve_path(args.predictions)
    output_dir = resolve_path(args.output_dir)
    predictions = load_predictions(predictions_path)

    if args.category != "all":
        predictions = [
            prediction
            for prediction in predictions
            if prediction["error_category"] == args.category
        ]

    predictions = predictions[:args.limit]

    for index, prediction in enumerate(predictions):
        category = prediction["error_category"]
        category_dir = output_dir / category
        question_id = prediction.get("question_id", index)
        filename = f"{index:04d}_{safe_filename(question_id)}.png"
        ground_truth_answers = prediction.get("ground_truth_answers", [""])

        visualize_prediction(
            image_path=str(resolve_path(prediction["image_path"])),
            predicted_bbox=prediction["predicted_bbox"],
            predicted_answer=prediction["predicted_answer"],
            question=prediction["question"],
            output_path=str(category_dir / filename),
            ground_truth_bbox=prediction["ground_truth_bbox"],
            ground_truth_answer=ground_truth_answers[0] if ground_truth_answers else ""
        )

    print(f"{len(predictions)} visualisasi disimpan di {output_dir}")

if __name__ == "__main__":
    main()
