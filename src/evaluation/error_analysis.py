from collections import Counter, defaultdict
from typing import Dict, List

def categorize_prediction(
    exact_match: float,
    anls: float,
    iou: float,
    iou_threshold: float = 0.5,
    anls_threshold: float = 0.5,
) -> str:
    text_correct = exact_match == 1.0
    location_correct = iou >= iou_threshold
    text_similar = anls >= anls_threshold

    if text_correct and location_correct:
        return "correct"
    if text_correct and not location_correct:
        return "text_correct_location_wrong"
    if not text_correct and location_correct:
        return "text_wrong_location_correct"
    if text_similar:
        return "partially_correct_text"
    return "both_wrong"

def analyze_errors(
    predictions: List[Dict],
    iou_threshold: float = 0.5,
    anls_threshold: float = 0.5,
):
    categorized_predictions = []
    category_counter = Counter()
    type_statistics = defaultdict(Counter)

    for prediction in predictions:
        category = categorize_prediction(
            exact_match=prediction["exact_match"],
            anls=prediction["anls"],
            iou=prediction["iou"],
            iou_threshold=iou_threshold,
            anls_threshold=anls_threshold,
        )
        enriched_prediction = {
            **prediction,
            "error_category": category,
        }
        categorized_predictions.append(enriched_prediction)
        category_counter[category] += 1

        question_type = prediction.get("question_type", "unknown")
        type_statistics[question_type][category] += 1

    total = len(categorized_predictions)
    category_percentages = {
        category: (
            count / total * 100
            if total > 0
            else 0.0
        )
        for category, count
        in category_counter.items()
    }

    summary = {
        "total_samples": total,
        "iou_threshold": iou_threshold,
        "anls_threshold": anls_threshold,
        "category_counts": dict(
            category_counter
        ),
        "category_percentages": (
            category_percentages
        ),
        "per_question_type": {
            question_type: dict(counts)
            for question_type, counts
            in type_statistics.items()
        },
    }

    return categorized_predictions, summary