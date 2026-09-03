import json
from pathlib import Path

def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)

def get_ocr_pages(payload):
    if payload.get("recognitionResults"):
        return payload["recognitionResults"]
    if payload.get("readResults"):
        return payload["readResults"]

    analyze_result = payload.get("analyzeResult", {})
    if analyze_result.get("readResults"):
        return analyze_result["readResults"]
    if analyze_result.get("pages"):
        return analyze_result["pages"]

    return []

def polygon_to_box(polygon):
    if not polygon:
        return None
    if isinstance(polygon[0], dict):
        x_values = [point["x"] for point in polygon]
        y_values = [point["y"] for point in polygon]
    elif isinstance(polygon[0], (list, tuple)):
        x_values = [point[0] for point in polygon]
        y_values = [point[1] for point in polygon]
    elif len(polygon) == 4:
        return [
            float(polygon[0]),
            float(polygon[1]),
            float(polygon[2]),
            float(polygon[3])
        ]
    else:
        x_values = polygon[0::2]
        y_values = polygon[1::2]

    return [
        float(min(x_values)),
        float(min(y_values)),
        float(max(x_values)),
        float(max(y_values))
    ]

def extract_page_words(page):
    words = []

    lines = page.get("lines", [])
    for line_index, line in enumerate(lines):
        for word_index, word in enumerate(line.get("words", [])):
            text = str(word.get("text") or word.get("content") or "").strip()
            polygon = word.get("boundingBox") or word.get("polygon") or []
            box = polygon_to_box(polygon)
            if text and box:
                words.append({
                    "text": text,
                    "box": box,
                    "line_index": line_index,
                    "word_index": word_index
                })

    if not words:
        for word_index, word in enumerate(page.get("words", [])):
            text = str(word.get("text") or word.get("content") or "").strip()
            polygon = word.get("boundingBox") or word.get("polygon") or []
            box = polygon_to_box(polygon)
            if text and box:
                words.append({
                    "text": text,
                    "box": box,
                    "line_index": None,
                    "word_index": word_index
                })

    return words

def load_ocr(path):
    payload = load_json(path)
    pages = get_ocr_pages(payload)
    if not pages:
        raise ValueError(f"Tidak menemukan halaman OCR: {path}")

    page = pages[0]
    width = page.get("width")
    height = page.get("height")
    unit = page.get("unit", "pixel")

    words = extract_page_words(page)

    return {
        "width": width,
        "height": height,
        "unit": unit,
        "words": words,
        "raw": payload
    }