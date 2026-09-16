import argparse
import json
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocess_dataset import preprocess_split

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline confidence-aware SP-DocVQA."
    )
    parser.add_argument(
        "--config",
        default="configs/layoutlmv3.yaml",
        help="Konfigurasi tunggal pipeline LayoutLMv3.",
    )
    return parser.parse_args()

def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path

def load_config(path):
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def validate_box(box, record_id):
    if len(box) != 4:
        raise ValueError(f"Panjang bbox tidak valid: {record_id}")
    x1, y1, x2, y2 = box
    if not all(0 <= coordinate <= 1000 for coordinate in box):
        raise ValueError(f"Bbox di luar rentang 0-1000: {record_id}")
    if x1 > x2 or y1 > y2:
        raise ValueError(f"Urutan bbox tidak valid: {record_id}")

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
            raise ValueError(f"Window tanpa word: {record_id}")
        if not len(words) == len(boxes) == len(confidence_labels):
            raise ValueError(f"Panjang feature word berbeda: {record_id}")

        for box in boxes:
            validate_box(box, record_id)

        candidates = record["candidate_spans"]
        if require_candidates and not candidates:
            raise ValueError(f"Window training tanpa kandidat: {record_id}")
        for candidate in candidates:
            start_word = candidate["start_word"]
            end_word = candidate["end_word"]
            if not 0 <= start_word <= end_word < len(words):
                raise ValueError(f"Indeks kandidat tidak valid: {record_id}")
            if candidate["weight"] <= 0:
                raise ValueError(f"Bobot kandidat tidak positif: {record_id}")
            validate_box(candidate["box_normalized"], record_id)

    print(f"Sanity check berhasil: {processed_path} ({len(records)} window)")

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
            raise FileNotFoundError(f"Input preprocessing tidak ditemukan: {path}")

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
            "name": "train_sample",
            "annotation_path": train_annotation,
            "image_dir": train_image_dir,
            "ocr_dir": train_ocr_dir,
            "output_path": resolve_path(data_config["train_path"]),
            "split": "train",
            "positive_only": True,
        },
        {
            "name": "val_sample",
            "annotation_path": val_annotation,
            "image_dir": val_image_dir,
            "ocr_dir": val_ocr_dir,
            "output_path": resolve_path(data_config["val_path"]),
            "split": "val",
            "positive_only": True,
        },
        {
            "name": "val_eval_sample",
            "annotation_path": val_annotation,
            "image_dir": val_image_dir,
            "ocr_dir": val_ocr_dir,
            "output_path": resolve_path(data_config["val_eval_path"]),
            "split": "val_eval",
            "positive_only": False,
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