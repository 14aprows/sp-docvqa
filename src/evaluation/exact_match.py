import re 

def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def exact_match(prediction: str, ground_truth: str) -> float:
    prediction = normalize_text(prediction)
    ground_truth = normalize_text(ground_truth)
    return float(prediction == ground_truth)