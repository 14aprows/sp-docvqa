import json
from pathlib import Path
from PIL import Image

from src.data.ocr_loader import load_ocr
from src.preprocessing.answer_matching import match_answer
from src.preprocessing.bbox_normalization import normalize_box
from src.preprocessing.sliding_window import create_word_windows

def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def get_records(payload):
    possible_keys = [
        "data",
        "questions",
        "annotations"
    ]

    for key in possible_keys:
        if isinstance(payload.get(key), list):
            return payload[key]
    raise KeyError(f"List anotasi tidak ditemukan. Field tersedia: {list(payload.keys())}")

def get_question(record):
    return str(record.get("question") or record.get("questions") or "").strip()

def get_answer(record):
    value = record.get("answer") or record.get("answers") or []
    if isinstance(value, str):
        value = [value]
    return [str(answer).strip() for answer in value if str(answer).strip()]

def get_question_types(record):
    value = record.get("question_types") or record.get("question_type") or record.get("questionTypes") or []
    if isinstance(value, str):
        value = [value]
    return [str(qtype).strip() for qtype in value if str(qtype).strip()]

def get_question_id(record, fallback):
    return str(record.get("questionId") or record.get("question_id") or fallback)

def get_doc_id(record):
    value = record.get("docId") or record.get("doc_id") or record.get("document_id") or record.get("image") or record.get("image_name") or ""
    return Path(str(value)).stem

def build_file_index(directory, patterns):
    directory = Path(directory)
    file_index = {}

    for pattern in patterns:
        for path in directory.rglob(pattern):
            file_index.setdefault(path.stem.lower(), path)

    return file_index

def get_candidate_names(record):
    candidates = [get_doc_id(record).lower()]
    image_name = record.get("image") or record.get("image_name")
    if image_name:
        candidates.append(Path(str(image_name)).stem.lower())
    return list(dict.fromkeys(candidates))

def find_file(file_index, record):
    candidates = get_candidate_names(record)
    for candidate in candidates:
        if candidate in file_index:
            return file_index[candidate]

    return None

def preprocess_split(
    project_root,
    annotation_path,
    image_dir,
    ocr_dir,
    output_path,
    split,
    max_words=300,
    stride=80,
    positive_only=True
):
    project_root = Path(project_root).resolve()
    annotation_path = Path(annotation_path)
    image_dir = Path(image_dir)
    ocr_dir = Path(ocr_dir)
    output_path = Path(output_path)

    annotation_payload = load_json(annotation_path)
    records = get_records(annotation_payload)
    image_index = build_file_index(image_dir, ["*.jpg", "*.jpeg", "*.png"])
    ocr_index = build_file_index(ocr_dir, ["*.json"])

    processed_records = []
    report = {
        "split": split,
        "total_questions": len(records),
        "missing_images": 0,
        "missing_ocr": 0,
        "empty_questions": 0,
        "empty_answers": 0,
        "empty_ocr": 0,
        "answer_not_matched": 0,
        "no_positive_windows": 0,
        "processed_questions": 0,
        "generated_windows": 0
    }

    error_examples = {
        "missing_images": [],
        "missing_ocr": [],
        "answer_not_matched": []
    }

    for record_index, record in enumerate(records):
        question_id = get_question_id(record, fallback=f"{split}-{record_index}")
        current_record_id = get_doc_id(record)
        question = get_question(record)
        answer_candidates = get_answer(record)
        if not question:
            report["empty_questions"] += 1
            continue
        if not answer_candidates:
            report["empty_answers"] += 1
            continue

        image_path = find_file(image_index, record)
        if image_path is None:
            report["missing_images"] += 1
            if len(error_examples["missing_images"]) < 10:
                error_examples["missing_images"].append({
                    "question_id": question_id,
                    "doc_id": current_record_id
                })
            continue

        ocr_path = find_file(ocr_index, record)
        if ocr_path is None:
            report["missing_ocr"] += 1
            if len(error_examples["missing_ocr"]) < 10:
                error_examples["missing_ocr"].append({
                    "question_id": question_id,
                    "doc_id": current_record_id
                })
            continue

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        ocr_data = load_ocr(ocr_path)
        ocr_words = ocr_data["words"]

        if not ocr_words:
            report["empty_ocr"] += 1
            continue

        match_result = match_answer(
            words=ocr_words,
            answer_candidates=answer_candidates
        )

        if not match_result["matched"]:
            report["answer_not_matched"] += 1
            if len(error_examples["answer_not_matched"]) < 10:
                error_examples["answer_not_matched"].append({
                    "question_id": question_id,
                    "doc_id": current_record_id,
                    "question": question,
                    "answers": answer_candidates
                })
            continue

        words = [word["text"] for word in ocr_words]
        normalized_boxes = [
            normalize_box(
                box=word["box"],
                image_width=image_width,
                image_height=image_height,
                target_size=1000
            )
            for word in ocr_words
        ]

        answer_start_word = min(match_result["word_indices"])
        answer_end_word = max(match_result["word_indices"])
        normalized_answer_box = normalize_box(
            box=match_result["box"],
            image_width=image_width,
            image_height=image_height,
            target_size=1000
        )

        windows = create_word_windows(
            words=words,
            boxes=normalized_boxes,
            answer_start_word=answer_start_word,
            answer_end_word=answer_end_word,
            max_words=max_words,
            stride=stride,
            positive_only=positive_only
        )

        if not windows:
            report["no_positive_windows"] += 1
            continue

        relative_image_path = image_path.resolve().relative_to(project_root).as_posix()

        for window in windows:
            processed_record = {
                "id": f"{question_id}_window_{window['window_index']}",
                "question_id": question_id,
                "doc_id": current_record_id,
                "split": split,
                "question_types": get_question_types(record),
                "image_path": relative_image_path,
                "image_width": image_width,
                "image_height": image_height,
                "question": question,
                "answers": answer_candidates,
                "matched_answer": match_result["answer"],
                "answer_box_pixel": match_result["box"],
                "answer_box_normalized": normalized_answer_box,
                "window_index": window["window_index"],
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "contains_answer": window["contains_answer"],
                "words": window["words"],
                "boxes": window["boxes"],
                "answer_start_word": window["answer_start_word"],
                "answer_end_word": window["answer_end_word"]
            }
            processed_records.append(processed_record)

        report["processed_questions"] += 1
        report["generated_windows"] += len(windows)

    if report["total_questions"] > 0:
        report["processed_percentage"] = report["processed_questions"] / report["total_questions"] * 100
    else:
        report["processed_percentage"] = 0.0

    output_payload = {
        "dataset": "SP-DocVQA",
        "split": split,
        "max_words": max_words,
        "stride": stride,
        "positive_only": positive_only,
        "report": report,
        "error_examples": error_examples,
        "data": processed_records
    }

    save_json(output_payload, output_path)

    return report