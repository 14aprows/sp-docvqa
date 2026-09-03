from src.evaluation.exact_match import normalize_text

def levenshtein_distance(first_text: str, second_text: str) -> int:
    if len(first_text) < len(second_text):
        first_text, second_text = second_text, first_text

    previous_row = list(range(len(second_text) + 1))
    for first_index, first_character in enumerate(first_text, start=1):
        current_row = [first_index]
        for second_index, second_character in enumerate(second_text, start=1):
            insertion = current_row[second_index - 1] + 1
            deletion = previous_row[second_index] + 1
            substitution = previous_row[second_index - 1] + (first_character != second_character)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    return previous_row[-1]

def anls_score(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    prediction = normalize_text(prediction)
    ground_truth = normalize_text(ground_truth)
    if prediction == ground_truth:
        return 1.0

    maximum_length = max(len(prediction), len(ground_truth))
    if maximum_length == 0:
        return 1.0

    edit_distance = levenshtein_distance(prediction, ground_truth)
    normalized_distance = edit_distance / maximum_length
    if normalized_distance >= threshold:
        return 0.0
    return 1.0 - normalized_distance
