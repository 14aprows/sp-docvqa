from difflib import SequenceMatcher

from src.preprocessing.bbox_normalization import merge_boxes
from src.preprocessing.text_normalization import compact_normalize, normalize_text

EXACT_MATCH_TYPES = {"normalized_exact", "compact_exact"}

def is_low_confidence(word):
    label = word.get("confidence_label")
    return label is not None and str(label).strip().lower() == "low"

def calculate_ocr_quality_score(words, low_confidence_penalty=0.20):
    if not words:
        return 0.0

    low_confidence_word_count = sum(is_low_confidence(word) for word in words)
    low_confidence_ratio = low_confidence_word_count / len(words)
    return max(0.0, 1.0 - low_confidence_penalty * low_confidence_ratio)

def create_character_map(words, normalizer, separator):
    combined_text = ""
    word_ranges = []

    for word_index, word in enumerate(words):
        normalized_word = normalizer(word["text"])
        if combined_text and separator:
            combined_text += separator

        start = len(combined_text)
        combined_text += normalized_word
        word_ranges.append((
            word_index,
            start,
            len(combined_text)
        ))

    return combined_text, word_ranges

def find_character_occurrences(text, query):
    start = 0
    while start <= len(text) - len(query):
        match_start = text.find(query, start)
        if match_start == -1:
            break
        yield match_start, match_start + len(query)
        start = match_start + 1

def character_span_to_word_span(word_ranges, character_start, character_end):
    overlapping_words = [
        word_index for word_index, word_start, word_end in word_ranges
        if word_end > character_start and word_start < character_end
    ]
    if not overlapping_words:
        return None

    first_word = overlapping_words[0]
    last_word = overlapping_words[-1]
    first_range = word_ranges[first_word]
    last_range = word_ranges[last_word]

    if first_range[1] != character_start or last_range[2] != character_end:
        return None

    return first_word, last_word

def create_candidate(
    answer,
    words,
    start_word,
    end_word,
    match_type,
    text_similarity,
    low_confidence_penalty
):
    current_words = words[start_word:end_word + 1]
    low_confidence_word_count = sum(is_low_confidence(word) for word in current_words)
    low_confidence_ratio = low_confidence_word_count / len(current_words)
    ocr_quality_score = calculate_ocr_quality_score(
        current_words,
        low_confidence_penalty=low_confidence_penalty
    )

    return {
        "answer": answer,
        "ocr_text": " ".join(str(word["text"]) for word in current_words),
        "start_word": start_word,
        "end_word": end_word,
        "word_indices": list(range(start_word, end_word + 1)),
        "box": merge_boxes([word["box"] for word in current_words]),
        "match_type": match_type,
        "text_similarity": float(text_similarity),
        "low_confidence_word_count": low_confidence_word_count,
        "low_confidence_ratio": low_confidence_ratio,
        "ocr_quality_score": ocr_quality_score,
        "weight": float(text_similarity * ocr_quality_score)
    }

def find_exact_candidates(words, answer, low_confidence_penalty):
    candidates = []
    matching_methods = (
        ("normalized_exact", normalize_text, " "),
        ("compact_exact", compact_normalize, "")
    )

    for match_type, normalizer, separator in matching_methods:
        normalized_answer = normalizer(answer)
        if not normalized_answer:
            continue

        ocr_text, word_ranges = create_character_map(
            words,
            normalizer,
            separator
        )

        for character_start, character_end in find_character_occurrences(ocr_text, normalized_answer):
            word_span = character_span_to_word_span(
                word_ranges,
                character_start,
                character_end
            )
            if word_span is None:
                continue

            candidates.append(create_candidate(
                answer=answer,
                words=words,
                start_word=word_span[0],
                end_word=word_span[1],
                match_type=match_type,
                text_similarity=1.0,
                low_confidence_penalty=low_confidence_penalty
            ))

    return candidates

def get_fuzzy_span_lengths(answer, span_tolerance, max_answer_words):
    answer_word_count = max(1, len(normalize_text(answer).split()))
    minimum_length = max(1, answer_word_count - span_tolerance)
    maximum_length = min(max_answer_words, answer_word_count + span_tolerance)
    return range(minimum_length, maximum_length + 1)

def find_fuzzy_candidates(
    words,
    answer,
    fuzzy_threshold,
    span_tolerance,
    max_answer_words,
    low_confidence_penalty
):
    normalized_answer = compact_normalize(answer)
    if not normalized_answer:
        return []

    candidates = []
    for span_length in get_fuzzy_span_lengths(
        answer,
        span_tolerance,
        max_answer_words
    ):
        for start_word in range(len(words) - span_length + 1):
            end_word = start_word + span_length - 1
            ocr_text = " ".join(
                str(word["text"]) for word in words[start_word:end_word + 1]
            )
            normalized_ocr = compact_normalize(ocr_text)
            if not normalized_ocr:
                continue

            similarity = SequenceMatcher(
                None,
                normalized_ocr,
                normalized_answer
            ).ratio()
            if similarity < fuzzy_threshold:
                continue

            candidates.append(
                create_candidate(
                    answer=answer,
                    words=words,
                    start_word=start_word,
                    end_word=end_word,
                    match_type="fuzzy",
                    text_similarity=similarity,
                    low_confidence_penalty=low_confidence_penalty
                )
            )

    return candidates

def candidate_priority(candidate):
    exact_rank = 0 if candidate["match_type"] in EXACT_MATCH_TYPES else 1
    normalized_rank = 0 if candidate["match_type"] == "normalized_exact" else 1
    return (
        exact_rank,
        -candidate["weight"],
        normalized_rank,
        -candidate["text_similarity"],
        candidate["start_word"],
        candidate["end_word"]
    )

def match_answers(
    words,
    answer_candidates,
    fuzzy_threshold=0.85,
    span_tolerance=2,
    max_answer_words=15,
    max_candidates=20,
    low_confidence_penalty=0.2
):
    if isinstance(answer_candidates, str):
        answer_candidates = [answer_candidates]

    candidates_by_span = {}
    for answer_value in answer_candidates:
        answer = str(answer_value).strip()
        if not compact_normalize(answer):
            continue

        answer_matches = find_exact_candidates(
            words,
            answer,
            low_confidence_penalty
        )
        if not answer_matches:
            answer_matches = find_fuzzy_candidates(
                words=words,
                answer=answer,
                fuzzy_threshold=fuzzy_threshold,
                span_tolerance=span_tolerance,
                max_answer_words=max_answer_words,
                low_confidence_penalty=low_confidence_penalty
            )

        for candidate in answer_matches:
            span_key = (candidate["start_word"], candidate["end_word"])
            previous_candidate = candidates_by_span.get(span_key)
            if (previous_candidate is None
                or candidate_priority(candidate) 
                < candidate_priority(previous_candidate)):
                candidates_by_span[span_key] = candidate

    candidates = sorted(candidates_by_span.values(), key=candidate_priority)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    return {
        "matched": bool(candidates),
        "candidate_count": len(candidates),
        "is_ambiguous": len(candidates) > 1,
        "best_candidate": candidates[0] if candidates else None,
        "candidates": candidates
    }