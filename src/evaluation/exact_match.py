from src.preprocessing.text_normalization import normalize_text

def exact_match(prediction: str, ground_truth: str) -> float:
    prediction = normalize_text(prediction)
    ground_truth = normalize_text(ground_truth)
    return float(prediction == ground_truth)