import torch 
import torch.nn.functional as F

def multi_span_loss(
    start_logits,
    end_logits,
    candidate_start_positions,
    candidate_end_positions,
    candidate_weights,
    candidate_mask
):
    candidate_mask = candidate_mask.bool()
    if torch.any(candidate_mask.sum(dim=1) == 0):
        raise ValueError("Setiap sampel harus memiliki minimal satu kandidat valid")

    sequence_length = start_logits.size(1)
    valid_positions = (
        candidate_start_positions.ge(0) &
        candidate_start_positions.lt(sequence_length) &
        candidate_end_positions.ge(candidate_start_positions) &
        candidate_end_positions.lt(sequence_length)
    )
    if torch.any(candidate_mask & ~valid_positions):
        raise ValueError("Posisi kandidat berada di luar sequence model")

    start_log_probabilities = F.log_softmax(start_logits, dim=-1)
    end_log_probabilities = F.log_softmax(end_logits, dim=-1)
    selected_start = torch.gather(
        start_log_probabilities,
        dim=-1,
        index=candidate_start_positions
    )
    selected_end = torch.gather(
        end_log_probabilities,
        dim=1,
        index=candidate_end_positions
    )

    masked_weights = candidate_weights.to(start_logits.dtype) * candidate_mask
    weight_sum = masked_weights.sum(dim=1, keepdim=True)
    if torch.any(weight_sum <= 0):
        raise ValueError("Jumlah bobot kandidat valid harus positif")
    
    normalized_weights = masked_weights / weight_sum

    candidate_log_probabilities = selected_start + selected_end + torch.log(normalized_weights.clamp_min(1e-12))

    marginal_log_probability = torch.logsumexp(
        candidate_log_probabilities,
        dim=1
    )
    return -marginal_log_probability.mean()