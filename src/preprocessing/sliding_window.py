def create_word_windows(
    words,
    boxes,
    answer_start_word,
    answer_end_word,
    max_words=300,
    stride=80,
    positive_only=True
):
    if len(words) != len(boxes):
        raise ValueError("Jumlah words dan boxes harus sama")
    if max_words <= 0:
        raise ValueError("max_words harus lebih besar dari 0")
    if stride <= 0:
        raise ValueError("stride harus lebih besar dari 0")
    if stride >= max_words:
        raise ValueError("stride harus lebih kecil dari max_words")
    if not words:
        return []
    if answer_start_word is not None and answer_end_word is not None:
        valid_answer_indices = 0 <= answer_start_word <= answer_end_word < len(words)
        if not valid_answer_indices:
            raise ValueError("Indeks jawaban tidak valid")

    windows = []
    step = max_words - stride

    window_start = 0
    window_index = 0
    while window_start < len(words):
        window_end = min(window_start + max_words, len(words))
        contains_answer = (
            answer_start_word is not None
            and answer_end_word is not None
            and answer_start_word >= window_start
            and answer_end_word < window_end
        )
        if contains_answer:
            relative_answer_start = answer_start_word - window_start
            relative_answer_end = answer_end_word - window_start
        else:
            relative_answer_start = None
            relative_answer_end = None

        if contains_answer or not positive_only:
            windows.append(
                {
                    "window_index": window_index,
                    "window_start": window_start,
                    "window_end": window_end,
                    "words": words[window_start:window_end],
                    "boxes": boxes[window_start:window_end],
                    "contains_answer": contains_answer,
                    "answer_start_word": relative_answer_start,
                    "answer_end_word": relative_answer_end,
                }
            )

        if window_end == len(words):
            break

        window_start += step
        window_index += 1

    return windows