import argparse
import json
import random
import shutil
from pathlib import Path

DEFAULT_SEED = 42
DEFAULT_TRAIN_SIZE = 500
DEFAULT_VAL_SIZE = 100

def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def save_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

def get_records_key(payload):
    possible_keys = [
        "data",
        "questions",
        "annotations"
    ]

    for key in possible_keys:
        if key in payload:
            if isinstance(payload[key], list):
                return key
    raise KeyError(f"List anotasi tidak ditemukan. Field tersedia: {list(payload.keys())}")

def get_doc_id(record):
    doc_id = (
        record.get("docId") or record.get("doc_id") or record.get("document_id") or record.get("image") or record.get("image_name")
    )
    if doc_id is None:
        raise KeyError(f"Doc ID tidak ditemukan: {record}")
    return Path(str(doc_id)).stem

def get_candidate_names(record):
    candidates = []
    doc_id = get_doc_id(record)
    candidates.append(doc_id.lower())

    image_name = record.get("image") or record.get("image_name")
    if image_name:
        candidates.append(Path(str(image_name)).stem.lower())
    return list(dict.fromkeys(candidates))

def build_file_index(directory, extensions):
    index = {}
    for extension in extensions:
        for path in directory.rglob(extension):
            index[path.name.lower()] = path
            index[path.stem.lower()] = path
    return index

def find_file(index, candidates):
    for candidate in candidates:
        candidate = candidate.lower()
        if candidate in index:
            return index[candidate]
    return None

def create_annotation_sample(
    annotation_path,
    sample_size,
    seed
):
    payload = load_json(annotation_path)
    records_key = get_records_key(payload)
    records = payload[records_key]
    random_generator = random.Random(seed)

    selected_records = random_generator.sample(
        records,
        min(sample_size, len(records))
    )
    sample_payload = payload.copy()
    sample_payload[records_key] = selected_records
    return sample_payload, selected_records

def copy_sample_resources(
    records,
    raw_image_dir,
    raw_ocr_dir,
    sample_image_dir,
    sample_ocr_dir
):
    sample_image_dir.mkdir(parents=True, exist_ok=True)
    sample_ocr_dir.mkdir(parents=True, exist_ok=True)

    image_index = build_file_index(
        raw_image_dir,
        ["*.png", "*.jpg", "*.jpeg"]
    )
    ocr_index = build_file_index(
        raw_ocr_dir,
        ["*.json"]
    )

    copied_images = set()
    copied_ocr = set()

    missing_images = []
    missing_ocr = []

    for record in records:
        doc_id = get_doc_id(record)
        candidates = get_candidate_names(record)

        image_path = find_file(
            image_index,
            candidates
        )
        ocr_path = find_file(
            ocr_index,
            candidates
        )

        if image_path is not None:
            destination = sample_image_dir / image_path.name
            if destination.name not in copied_images:
                shutil.copy2(image_path, destination)
                copied_images.add(destination.name)
        else:
            missing_images.append(doc_id)

        if ocr_path is not None:
            destination = sample_ocr_dir / ocr_path.name
            if destination.name not in copied_ocr:
                shutil.copy2(ocr_path, destination)
                copied_ocr.add(destination.name)
        else:
            missing_ocr.append(doc_id)

    return {
        "copied_images": len(copied_images),
        "copied_ocr": len(copied_ocr),
        "missing_images": sorted(set(missing_images)),
        "missing_ocr": sorted(set(missing_ocr))
    }

def process_split(
    project_root,
    split,
    annotation_filename,
    sample_size,
    seed
):
    raw_dir = project_root / "data" / "raw"
    sample_dir = project_root / "data" / "samples"

    annotation_path = raw_dir / "annotations" / annotation_filename
    output_annotation_path = sample_dir / "annotations" / f"{split}_sample.json"

    sample_payload, selected_records = create_annotation_sample(
        annotation_path=annotation_path,
        sample_size=sample_size,
        seed=seed
    )
    save_json(sample_payload, output_annotation_path)

    copy_report = copy_sample_resources(
        records=selected_records,
        raw_image_dir=raw_dir / "images",
        raw_ocr_dir=raw_dir / "ocr",
        sample_image_dir=sample_dir / "images" / split,
        sample_ocr_dir=sample_dir / "ocr" / split
    )

    print(f"\nSplit: {split}")
    print(f"Questions: {len(selected_records)}")
    print(f"Images copied: {copy_report['copied_images']}")
    print(f"OCR copied: {copy_report['copied_ocr']}")
    print(f"Missing images: {len(copy_report['missing_images'])}")
    print(f"Missing OCR: {len(copy_report['missing_ocr'])}")
    print(f"Annotation: {output_annotation_path}")

    return {
        "split": split,
        "sample_questions": len(selected_records),
        **copy_report
    }

def main():
    parser = argparse.ArgumentParser(
        description="Membuat subset sample SP-DocVQA"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_SIZE
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=DEFAULT_VAL_SIZE
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    train_report = process_split(
        project_root=project_root,
        split="train",
        annotation_filename="train_v1.0_withQT.json",
        sample_size=args.train_size,
        seed=args.seed
    )
    val_report = process_split(
        project_root=project_root,
        split="val",
        annotation_filename="val_v1.0_withQT.json",
        sample_size=args.val_size,
        seed=args.seed
    )

    report = {
        "seed": args.seed,
        "train": train_report,
        "validation": val_report
    }
    report_path = project_root / "data" / "samples" / "sample_report.json"
    save_json(report, report_path)

    print("\nSample selesai dibuat.")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()
