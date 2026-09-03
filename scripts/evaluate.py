import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import evaluate_prediction
from src.evaluation.iou import union_boxes
from src.inference.predictor import LayoutLMv3QAPredictor
from src.models.model import build_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluasi checkpoint LayoutLMv3 untuk SP-DocVQA."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/layoutlmv3_spdocvqa/best"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/val_sample.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation/layoutlmv3_spdocvqa"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512
    )
    parser.add_argument(
        "--max-answer-tokens",
        type=int,
        default=30
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None
    )

    return parser.parse_args()

def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT / path

def load_records(json_path: Path) -> List[Dict]:
    if not json_path.exists():
        raise FileNotFoundError(f"Data evaluasi tidak ditemukan: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "data" in data:
        return data["data"]

    raise ValueError("Format JSON tidak dikenali.")

def get_ground_truth_answers(record: Dict) -> List[str]:
    answers = record.get("answers")

    if isinstance(answers, list) and answers:
        return [str(answer) for answer in answers]

    return [str(record.get("answer", ""))]

def get_ground_truth_bbox(record: Dict) -> List[int]:
    start_word = int(record["answer_start_word"])
    end_word = int(record["answer_end_word"])
    answer_boxes = record["boxes"][start_word:end_word + 1]

    return union_boxes(answer_boxes)

def get_question_type(record: Dict) -> str:
    question_type = record.get(
        "question_type",
        record.get("question_types", "unknown")
    )

    if isinstance(question_type, list):
        return " | ".join(str(item) for item in question_type)

    return str(question_type)

def calculate_average(items: List[Dict], metric_name: str) -> float:
    if not items:
        return 0.0

    return sum(item[metric_name] for item in items) / len(items)

def main():
    args = parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    data_path = resolve_path(args.data)
    output_dir = resolve_path(args.output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    records = load_records(data_path)

    if args.max_samples is not None:
        records = records[:args.max_samples]

    if not records:
        raise ValueError("Tidak ada data yang dievaluasi.")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Data:", data_path)
    print("Jumlah sampel:", len(records))

    processor = AutoProcessor.from_pretrained(
        checkpoint_path,
        apply_ocr=False
    )
    model = build_model(str(checkpoint_path))
    predictor = LayoutLMv3QAPredictor(
        model=model,
        processor=processor,
        device=device,
        max_length=args.max_length,
        max_answer_tokens=args.max_answer_tokens
    )

    predictions = []
    metrics_by_type = defaultdict(list)

    for record in tqdm(records, desc="Evaluasi"):
        prediction_record = dict(record)
        prediction_record["image_path"] = str(resolve_path(record["image_path"]))
        prediction = predictor.predict(prediction_record)

        ground_truth_answers = get_ground_truth_answers(record)
        ground_truth_bbox = get_ground_truth_bbox(record)
        metrics = evaluate_prediction(
            predicted_answer=prediction["answer"],
            ground_truth_answers=ground_truth_answers,
            predicted_bbox=prediction["bbox"],
            ground_truth_bbox=ground_truth_bbox
        )
        question_type = get_question_type(record)

        result = {
            "question_id": record.get("question_id"),
            "question": record["question"],
            "question_type": question_type,
            "ground_truth_answers": ground_truth_answers,
            "predicted_answer": prediction["answer"],
            "ground_truth_bbox": ground_truth_bbox,
            "predicted_bbox": prediction["bbox"],
            "predicted_start_word": prediction["start_word"],
            "predicted_end_word": prediction["end_word"],
            "span_score": prediction["span_score"],
            "exact_match": metrics["exact_match"],
            "anls": metrics["anls"],
            "iou": metrics["iou"],
            "image_path": record["image_path"]
        }

        predictions.append(result)
        metrics_by_type[question_type].append(metrics)

    overall = {
        "sample_count": len(predictions),
        "exact_match": calculate_average(predictions, "exact_match"),
        "anls": calculate_average(predictions, "anls"),
        "iou": calculate_average(predictions, "iou")
    }
    per_question_type = {
        question_type: {
            "sample_count": len(values),
            "exact_match": calculate_average(values, "exact_match"),
            "anls": calculate_average(values, "anls"),
            "iou": calculate_average(values, "iou")
        }
        for question_type, values in metrics_by_type.items()
    }
    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(data_path),
        "overall": overall,
        "per_question_type": per_question_type
    }

    predictions_path = output_dir / "predictions.json"
    summary_path = output_dir / "evaluation_summary.json"

    with predictions_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=2, ensure_ascii=False)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("HASIL EVALUASI")
    print("=" * 60)
    print("Jumlah sampel:", overall["sample_count"])
    print("Exact Match:", f'{overall["exact_match"]:.4f}')
    print("ANLS:", f'{overall["anls"]:.4f}')
    print("IoU:", f'{overall["iou"]:.4f}')
    print("=" * 60)
    print("Predictions:", predictions_path)
    print("Summary:", summary_path)

if __name__ == "__main__":
    main()
