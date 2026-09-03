from typing import Dict, List

from src.evaluation.anls import anls_score
from src.evaluation.exact_match import exact_match
from src.evaluation.iou import bbox_iou

def evaluate_prediction(
    predicted_answer: str,
    ground_truth_answers: List[str],
    predicted_bbox: List[int],
    ground_truth_bbox: List[int]
) -> Dict[str, float]:
    if not ground_truth_answers:
        ground_truth_answers = [""]

    exact_match_scores = [
        exact_match(predicted_answer, ground_truth)
        for ground_truth in ground_truth_answers
    ]
    anls_scores = [
        anls_score(predicted_answer, ground_truth)
        for ground_truth in ground_truth_answers
    ]

    return {
        "exact_match": max(exact_match_scores),
        "anls": max(anls_scores),
        "iou": bbox_iou(predicted_bbox, ground_truth_bbox)
    }
