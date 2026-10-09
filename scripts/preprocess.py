import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocess_dataset import preprocess_split
from src.utils.config import load_config, resolve_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file",
    )
    return parser.parse_args()

def validate_box(box, record_id):
    if len(box) != 4:
        raise ValueError(f"Box must have 4 values: {record_id}")
    x1, y1, x2, y2 = box
    if not all(0 <= coordinate <= 1000 for coordinate in box):
        raise ValueError(f"Box values must be between 0 and 1000: {record_id}")
    if x1 > x2 or y1 > y2:
        raise ValueError(f"Box coordinates are in the wrong order: {record_id}")

def sanity_check(processed_path, require_candidates):
    with Path(processed_path).open("r", encoding="utf-8") as file:
        records = json.load(file)["data"]

    seen_ids = set()
    for record in records:
        record_id = record["id"]
        if record_id in seen_ids:
            raise ValueError(f"Duplicate ID: {record_id}")
        seen_ids.add(record_id)

        words = record["words"]
        boxes = record["boxes"]
        confidence_labels = record["confidence_labels"]
        if not words:
            raise ValueError(f"Window has no words: {record_id}")
        if not len(words) == len(boxes) == len(confidence_labels):
            raise ValueError(f"Word feature lengths do not match: {record_id}")

        for box in boxes:
            validate_box(box, record_id)

        candidates = record["candidate_spans"]
        if require_candidates and not candidates:
            raise ValueError(f"Training window has no candidates: {record_id}")
        for candidate in candidates:
            start_word = candidate["start_word"]
            end_word = candidate["end_word"]
            if not 0 <= start_word <= end_word < len(words):
                raise ValueError(f"Candidate indices are invalid: {record_id}")
            if candidate["weight"] <= 0:
                raise ValueError(f"Candidate weight must be positive: {record_id}")
            validate_box(candidate["box_normalized"], record_id)

    print(f"Sanity check passed: {processed_path} ({len(records)} windows)")

def print_report(name, report):
    print(f"\n{name}")
    for key, value in report.items():
        formatted = f"{value:.2f}" if isinstance(value, float) else value
        print(f"  {key}: {formatted}")

def main():
    config = load_config(parse_args().config)
    data_config = config["data"]
    preprocessing_config = config["preprocessing"]

    train_annotation = resolve_path(data_config["train_annotation_path"])
    val_annotation = resolve_path(data_config["val_annotation_path"])
    train_image_dir = resolve_path(data_config["train_image_dir"])
    val_image_dir = resolve_path(data_config["val_image_dir"])
    train_ocr_dir = resolve_path(data_config["train_ocr_dir"])
    val_ocr_dir = resolve_path(data_config["val_ocr_dir"])

    required_paths = (
        train_annotation,
        val_annotation,
        train_image_dir,
        val_image_dir,
        train_ocr_dir,
        val_ocr_dir,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Preprocessing input not found: {path}")

    common_arguments = {
        "project_root": ROOT,
        "max_words": preprocessing_config["max_words"],
        "stride": preprocessing_config["stride"],
        "fuzzy_threshold": preprocessing_config["fuzzy_threshold"],
        "span_tolerance": preprocessing_config["span_tolerance"],
        "max_answer_words": preprocessing_config["max_answer_words"],
        "max_candidates": preprocessing_config["max_candidates"],
        "low_confidence_penalty": preprocessing_config["low_confidence_penalty"],
    }
    jobs = (
        {
            "name": "train",
            "annotation_path": train_annotation,
            "image_dir": train_image_dir,
            "ocr_dir": train_ocr_dir,
            "output_path": resolve_path(data_config["train_path"]),
            "split": "train",
            "positive_only": preprocessing_config["train_positive_only"]
        },
        {
            "name": "val",
            "annotation_path": val_annotation,
            "image_dir": val_image_dir,
            "ocr_dir": val_ocr_dir,
            "output_path": resolve_path(data_config["val_path"]),
            "split": "val",
            "positive_only": preprocessing_config["val_positive_only"]
        },
        {
            "name": "val_eval",
            "annotation_path": val_annotation,
            "image_dir": val_image_dir,
            "ocr_dir": val_ocr_dir,
            "output_path": resolve_path(data_config["val_eval_path"]),
            "split": "val_eval",
            "positive_only": preprocessing_config["val_eval_positive_only"],
        },
    )

    for job in jobs:
        name = job.pop("name")
        output_path = job["output_path"]
        report = preprocess_split(**common_arguments, **job)
        print_report(name, report)
        sanity_check(output_path, require_candidates=job["positive_only"])

if __name__ == "__main__":
    main()