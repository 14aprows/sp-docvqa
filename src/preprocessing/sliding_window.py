def validate_candidates(candidates, word_count):
    for candidate in candidates:
        start_word = candidate["start_word"]
        end_word = candidate["end_word"]
        if not 0 <= start_word <= end_word < word_count:
            raise ValueError(f"Indeks kandidat tidak valid: start={start_word}, end={end_word}, jumlah_word={word_count}")

def create_word_windows(
    words,
    boxes,
    confidence_labels,
    answer_candidates,
    max_words=300,
    stride=80,
    positive_only=True
):
    if len(words) != len(boxes):
        raise ValueError("Jumlah words dan boxes harus sama")
    if len(words) != len(confidence_labels):
        raise ValueError("Jumlah words dan confidence_labels harus sama")
    if max_words <= 0:
        raise ValueError("max_words harus lebih besar dari 0")
    if stride <= 0:
        raise ValueError("stride harus lebih besar dari 0")
    if stride >= max_words:
        raise ValueError("stride harus lebih kecil dari max_words")
    if not words:
        return []

    validate_candidates(answer_candidates, len(words))

    windows = []
    step = max_words - stride
    window_start = 0
    window_index = 0

    while window_start < len(words):
        window_end = min(window_start + max_words, len(words))
        window_candidates = []

        for candidate in answer_candidates:
            candidate_start = candidate["start_word"]
            candidate_end = candidate["end_word"]
            if candidate_start < window_start or candidate_end >= window_end:
                continue

            local_candidate = dict(candidate)
            local_candidate["global_start_word"] = candidate_start
            local_candidate["global_end_word"] = candidate_end
            local_candidate["start_word"] = candidate_start - window_start
            local_candidate["end_word"] = candidate_end - window_start
            local_candidate["word_indices"] = list(range(
                local_candidate["start_word"],
                local_candidate["end_word"] + 1,
            ))
            window_candidates.append(local_candidate)

        contains_answer = bool(window_candidates)
        if contains_answer or not positive_only:
            primary_candidate = window_candidates[0] if window_candidates else None
            windows.append({
                "window_index": window_index,
                "window_start": window_start,
                "window_end": window_end,
                "words": words[window_start:window_end],
                "boxes": boxes[window_start:window_end],
                "confidence_labels": confidence_labels[window_start:window_end],
                "contains_answer": contains_answer,
                "answer_start_word": primary_candidate["start_word"] if primary_candidate else None,
                "answer_end_word": primary_candidate["end_word"] if primary_candidate else None,
                "candidate_spans": window_candidates
            })

        if window_end == len(words):
            break

        window_start += step
        window_index += 1

    return windows