from src.preprocessing.text_normalization import compact_normalize
from src.preprocessing.bbox_normalization import merge_boxes

def create_ocr_character_map(words):
    combined_text = ""
    word_ranges = []
    for index, word in enumerate(words):
        normalized_word = compact_normalize(word["text"])
        start = len(combined_text)
        combined_text += normalized_word
        end = len(combined_text)
        word_ranges.append({
            "word_index": index,
            "start": start,
            "end": end
        })

    return combined_text, word_ranges

def find_answer_word_indices(
    words,
    answer
):
    normalized_answer = compact_normalize(answer)
    if not normalized_answer:
        return []

    combined_text, word_ranges = (create_ocr_character_map(words))
    answer_start = combined_text.find(normalized_answer)
    if answer_start == -1:
        return []

    answer_end = answer_start + len(normalized_answer)

    matched_indices = []
    for word_range in word_ranges:
        overlaps = (
            word_range["end"] > answer_start
            and word_range["start"] < answer_end
        )
        if overlaps:
            matched_indices.append(word_range["word_index"])

    return matched_indices

def match_answer(
    words,
    answer_candidates
):
    if isinstance(answer_candidates, str):
        answer_candidates = [answer_candidates]

    for answer in answer_candidates:
        indices = find_answer_word_indices(words,answer)
        if not indices:
            continue\
            
        matched_words = [words[index] for index in indices]
        answer_box = merge_boxes([
            word["box"]
            for word in matched_words
        ])

        return {
            "matched": True,
            "answer": answer,
            "word_indices": indices,
            "words": matched_words,
            "box": answer_box
        }

    return {
        "matched": False,
        "answer": None,
        "word_indices": [],
        "words": [],
        "box": None
    }