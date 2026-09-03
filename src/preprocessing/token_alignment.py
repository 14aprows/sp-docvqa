from typing import List, Optional, Tuple

def find_token_span(
    word_ids: List[Optional[int]],
    sequence_ids: List[Optional[int]],
    answer_start_word: int,
    answer_end_word: int
) -> Tuple[Optional[int], Optional[int]]:
    if len(word_ids) != len(sequence_ids):
        raise ValueError("Jumlah word_ids dan sequence_ids harus sama")
    if answer_start_word < 0:
        raise ValueError("answer_start_word tidak boleh negatif")
    if answer_end_word < answer_start_word:
        raise ValueError("answer_end_word harus lebih besar atau sama dengan answer_start_word")

    start_candidates = []
    end_candidates = []

    for token_index, (word_id, sequence_id) in enumerate(zip(word_ids, sequence_ids)):
        if sequence_id != 1:
            continue
        if word_id == answer_start_word:
            start_candidates.append(token_index)
        if word_id == answer_end_word:
            end_candidates.append(token_index)

    if not start_candidates or not end_candidates:
        return None, None

    start_token = min(start_candidates)
    end_token = max(end_candidates)

    if start_token > end_token:
        return None, None

    return start_token, end_token