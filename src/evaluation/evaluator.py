from src.evaluation.anls import anls_score
from src.evaluation.exact_match import exact_match
from src.evaluation.iou import bbox_iou

def evaluate_prediction(
    predicted_answer,
    ground_truth_answers,
    predicted_bbox,
    pseudo_ground_truth_bboxes,
):
    ground_truth_answers = ground_truth_answers or [""]
    pseudo_ground_truth_bboxes = pseudo_ground_truth_bboxes or [[0, 0, 0, 0]]

    exact_match_score = max(exact_match(predicted_answer, answer) for answer in ground_truth_answers)
    anls = max(anls_score(predicted_answer, answer) for answer in ground_truth_answers)
    iou = max(bbox_iou(predicted_bbox, candidate_box) for candidate_box in pseudo_ground_truth_bboxes)

    return {
        "exact_match": exact_match_score,
        "anls": anls,
        "iou": iou,
        "iou_at_0_5": float(iou >= 0.5),
        "iou_at_0_75": float(iou >= 0.75),
        "joint_anls_iou": anls * iou,
    }

def summarize_metrics(results):
    if not results:
        return {
            "question_count": 0,
            "exact_match": 0.0,
            "anls": 0.0,
            "mean_iou": 0.0,
            "iou_at_0_5": 0.0,
            "iou_at_0_75": 0.0,
            "joint_anls_iou": 0.0,
        }

    return {
        "question_count": len(results),
        "exact_match": sum(item["exact_match"] for item in results) / len(results),
        "anls": sum(item["anls"] for item in results) / len(results),
        "mean_iou": sum(item["iou"] for item in results) / len(results),
        "iou_at_0_5": sum(item["iou_at_0_5"] for item in results) / len(results),
        "iou_at_0_75": sum(item["iou_at_0_75"] for item in results) / len(results),
        "joint_anls_iou": (
            sum(item["joint_anls_iou"] for item in results) / len(results)
        ),
    }