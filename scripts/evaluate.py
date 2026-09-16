import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import evaluate_prediction, summarize_metrics
from src.inference.predictor import LayoutLMv3QAPredictor
from src.models.model import build_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/layoutlmv3.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    return parser.parse_args()

def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path

def load_config(path_value):
    with resolve_path(path_value).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def load_records(path):
    if not path.exists():
        raise FileNotFoundError(f"Data evaluasi tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Format data evaluasi tidak valid: {path}")

def get_question_types(record):
    values = record.get("question_types") or ["unknown"]
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values] or ["unknown"]

def unique_boxes(boxes):
    unique = []
    seen = set()
    for box in boxes:
        key = tuple(box)
        if key not in seen:
            seen.add(key)
            unique.append(box)
    return unique

def get_pseudo_ground_truth_boxes(records):
    boxes = []
    for record in records:
        boxes.extend(record.get("candidate_pseudo_ground_truth_boxes_normalized", []))
        if not record.get("candidate_pseudo_ground_truth_boxes_normalized"):
            boxes.extend(
                candidate["box_normalized"]
                for candidate in record.get("candidate_spans", [])
            )
    return unique_boxes(boxes)

def select_best_prediction(predictions):
    return max(
        predictions,
        key=lambda prediction: (
            prediction["span_score"]
            if prediction["span_score"] is not None
            else float("-inf")
        ),
    )

def main():
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    inference_config = config["inference"]
    evaluation_config = config["evaluation"]

    checkpoint_path = resolve_path(
        args.checkpoint or evaluation_config["checkpoint_path"]
    )
    data_path = resolve_path(args.data or config["data"]["val_eval_path"])
    output_dir = resolve_path(args.output_dir or evaluation_config["output_dir"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    grouped_records = defaultdict(list)
    for record in load_records(data_path):
        grouped_records[str(record["question_id"])].append(record)
    if args.max_questions is not None:
        grouped_records = dict(list(grouped_records.items())[:args.max_questions])
    if not grouped_records:
        raise ValueError("Tidak ada pertanyaan untuk dievaluasi.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        checkpoint_path,
        apply_ocr=False,
        local_files_only=True,
    )
    model = build_model(checkpoint_path, map_location=device)
    predictor = LayoutLMv3QAPredictor(
        model=model,
        processor=processor,
        device=device,
        max_length=model_config["max_length"],
        max_answer_tokens=inference_config["max_answer_tokens"],
        top_k=inference_config["top_k"],
    )

    predictions = []
    results_by_type = defaultdict(list)
    results_by_ambiguity = defaultdict(list)
    for question_id, question_records in tqdm(
        grouped_records.items(),
        desc="Evaluasi per pertanyaan",
    ):
        window_predictions = []
        for record in question_records:
            prediction_record = dict(record)
            prediction_record["image_path"] = str(resolve_path(record["image_path"]))
            window_predictions.append(predictor.predict(prediction_record))

        best_prediction = select_best_prediction(window_predictions)
        representative = question_records[0]
        pseudo_ground_truth_boxes = get_pseudo_ground_truth_boxes(question_records)
        metrics = evaluate_prediction(
            predicted_answer=best_prediction["answer"],
            ground_truth_answers=representative.get("answers", []),
            predicted_bbox=best_prediction["bbox_normalized"],
            pseudo_ground_truth_bboxes=pseudo_ground_truth_boxes,
        )
        question_types = get_question_types(representative)
        is_ambiguous = bool(representative.get("is_ambiguous"))

        result = {
            "question_id": question_id,
            "question": representative["question"],
            "question_types": question_types,
            "image_path": representative["image_path"],
            "ground_truth_answers": representative.get("answers", []),
            "pseudo_ground_truth_boxes_normalized": pseudo_ground_truth_boxes,
            "is_ambiguous": is_ambiguous,
            "evaluated_window_count": len(question_records),
            "predicted_answer": best_prediction["answer"],
            "predicted_bbox_normalized": best_prediction["bbox_normalized"],
            "predicted_start_token": best_prediction["start_token"],
            "predicted_end_token": best_prediction["end_token"],
            "predicted_start_word": best_prediction["start_word"],
            "predicted_end_word": best_prediction["end_word"],
            "predicted_global_start_word": best_prediction["global_start_word"],
            "predicted_global_end_word": best_prediction["global_end_word"],
            "selected_window_index": best_prediction["window_index"],
            "span_score": best_prediction["span_score"],
            "gate_values": best_prediction["gate_values"],
            **metrics,
        }
        predictions.append(result)
        for question_type in question_types:
            results_by_type[question_type].append(result)
        ambiguity_group = "ambiguous" if is_ambiguous else "non_ambiguous"
        results_by_ambiguity[ambiguity_group].append(result)

    summary = {
        "pipeline": config["pipeline"]["name"],
        "checkpoint": str(checkpoint_path),
        "dataset": str(data_path),
        "overall": summarize_metrics(predictions),
        "per_question_type": {
            question_type: summarize_metrics(results)
            for question_type, results in sorted(results_by_type.items())
        },
        "ambiguity_statistics": {
            group: summarize_metrics(results_by_ambiguity[group])
            for group in ("ambiguous", "non_ambiguous")
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions_per_question.json"
    summary_path = output_dir / "evaluation_summary.json"
    with predictions_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=2, ensure_ascii=False)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    print("Predictions:", predictions_path)
    print("Summary:", summary_path)

if __name__ == "__main__":
    main()