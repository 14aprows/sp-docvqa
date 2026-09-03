import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocess_dataset import preprocess_split
from src.preprocessing.text_normalization import compact_normalize

TRAIN_ANNOTATION_PATH = ROOT / "data" / "samples" / "annotations" / "train_sample.json"
VAL_ANNOTATION_PATH = ROOT / "data" / "samples" / "annotations" / "val_sample.json"

TRAIN_IMAGE_DIR = ROOT / "data" / "samples" / "images" / "train"
VAL_IMAGE_DIR = ROOT / "data" / "samples" / "images" / "val"

TRAIN_OCR_DIR = ROOT / "data" / "samples" / "ocr" / "train"
VAL_OCR_DIR = ROOT / "data" / "samples" / "ocr" / "val"

TRAIN_OUTPUT_PATH = ROOT / "data" / "processed" / "train_sample.json"
VAL_OUTPUT_PATH = ROOT / "data" / "processed" / "val_sample.json"

def print_report(split_name, report):
    print("\n" + "=" * 50)
    print(f"REPORT {split_name.upper()}")
    print("=" * 50)

    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

def sanity_check(processed_path):
    with Path(processed_path).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    records = payload["data"]
    seen_ids = set()

    for record in records:
        record_id = record["id"]

        if record_id in seen_ids:
            raise ValueError(f"Duplicate ID: {record_id}")

        seen_ids.add(record_id)

        words = record["words"]
        boxes = record["boxes"]

        if len(words) != len(boxes):
            raise ValueError(f"Words-boxes mismatch: {record_id}")

        if not words:
            raise ValueError(f"Empty words: {record_id}")

        for box in boxes:
            if len(box) != 4:
                raise ValueError(f"Invalid bbox length: {record_id}")

            x1, y1, x2, y2 = box

            if not all(0 <= coordinate <= 1000 for coordinate in box):
                raise ValueError(f"Bbox outside 0-1000: {record_id}")

            if x1 > x2 or y1 > y2:
                raise ValueError(f"Invalid bbox order: {record_id}")

        if record["contains_answer"]:
            start = record["answer_start_word"]
            end = record["answer_end_word"]

            if start is None or end is None:
                raise ValueError(f"Missing answer index: {record_id}")

            if not 0 <= start <= end < len(words):
                raise ValueError(f"Invalid answer index: {record_id}")

            reconstructed_answer = "".join(words[start:end + 1])
            expected_answer = record["matched_answer"]

            if compact_normalize(expected_answer) not in compact_normalize(reconstructed_answer):
                raise ValueError(
                    f"Answer mismatch: {record_id} | "
                    f"{reconstructed_answer} != {expected_answer}"
                )

        image_path = ROOT / record["image_path"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image tidak ditemukan: {image_path}")

    print(f"Sanity check berhasil: {len(records)} records")

def main():
    required_paths = [
        TRAIN_ANNOTATION_PATH,
        VAL_ANNOTATION_PATH,
        TRAIN_IMAGE_DIR,
        VAL_IMAGE_DIR,
        TRAIN_OCR_DIR,
        VAL_OCR_DIR
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Path tidak ditemukan: {path}")

    print("Memulai preprocessing train...")

    train_report = preprocess_split(
        project_root=ROOT,
        annotation_path=TRAIN_ANNOTATION_PATH,
        image_dir=TRAIN_IMAGE_DIR,
        ocr_dir=TRAIN_OCR_DIR,
        output_path=TRAIN_OUTPUT_PATH,
        split="train",
        max_words=300,
        stride=80,
        positive_only=True
    )

    print_report("train", train_report)

    print("\nMemulai preprocessing validation...")

    val_report = preprocess_split(
        project_root=ROOT,
        annotation_path=VAL_ANNOTATION_PATH,
        image_dir=VAL_IMAGE_DIR,
        ocr_dir=VAL_OCR_DIR,
        output_path=VAL_OUTPUT_PATH,
        split="val",
        max_words=300,
        stride=80,
        positive_only=True
    )

    print_report("validation", val_report)

    print("\nMenjalankan sanity check train...")
    sanity_check(TRAIN_OUTPUT_PATH)

    print("\nMenjalankan sanity check validation...")
    sanity_check(VAL_OUTPUT_PATH)

    print("\n" + "=" * 50)
    print("PREPROCESSING SELESAI")
    print("=" * 50)
    print(f"Train output: {TRAIN_OUTPUT_PATH}")
    print(f"Validation output: {VAL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
