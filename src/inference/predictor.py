from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import build_token_level_features
from src.evaluation.iou import union_boxes

class LayoutLMv3QAPredictor:
    def __init__(
        self,
        model,
        processor,
        device,
        max_length: int = 512,
        max_answer_tokens=30,
        top_k=20
    ):
        self.model = model.to(device)
        self.processor = processor
        self.device = device
        self.max_length = max_length
        self.max_answer_tokens = max_answer_tokens
        self.top_k = top_k
        self.model.eval()

    def _find_best_span(
        self,
        start_logits,
        end_logits,
        document_mask
    ):
        valid_positions = document_mask.bool()
        if not valid_positions.any():
            return None

        start_logits = start_logits.masked_fill(~valid_positions, -10000.0)
        end_logits = end_logits.masked_fill(~valid_positions, -10000.0)
        start_log_probabilities = F.log_softmax(start_logits, dim=-1)
        end_log_probabilities = F.log_softmax(end_logits, dim=-1)

        candidate_count = min(self.top_k, int(valid_positions.sum().item()))
        start_candidates = torch.topk(
            start_log_probabilities,
            k=candidate_count
        ).indices.tolist()
        end_candidates = torch.topk(
            end_log_probabilities,
            k=candidate_count
        ).indices.tolist()

        best_span = None
        best_score = float("-inf")
        for start_token in start_candidates:
            for end_token in end_candidates:
                span_length = end_token - start_token + 1
                if end_token < start_token or span_length > self.max_answer_tokens:
                    continue

                score = (start_log_probabilities[start_token] + end_log_probabilities[end_token]).item()
                if score > best_score:
                    best_score = score
                    best_span = (start_token, end_token, best_score)

        return best_span

    @torch.no_grad()
    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        image_path = Path(record["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

        words = record["words"]
        boxes = record["boxes"]
        confidence_labels = record["confidence_labels"]
        if not len(words) == len(boxes) == len(confidence_labels):
            record_id = record.get("id")
            raise ValueError(f"Feature word tidak sejajar: id={record_id}")

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        encoding = self.processor(
            image,
            str(record["question"]),
            words,
            boxes=boxes,
            truncation="only_second",
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        sequence_ids = encoding.sequence_ids(batch_index=0)
        confidence_ids, question_mask, document_mask = build_token_level_features(
            word_ids=word_ids,
            sequence_ids=sequence_ids,
            confidence_labels=confidence_labels,
            attention_mask=encoding["attention_mask"][0]
        )

        model_inputs = {
            key: value.to(self.device)
            for key, value in encoding.items()
            if isinstance(value, torch.Tensor)
        }

        model_inputs.update({
            "confidence_ids": confidence_ids.unsqueeze(0).to(self.device),
            "question_mask": question_mask.unsqueeze(0).to(self.device),
            "document_mask": document_mask.unsqueeze(0).to(self.device)
        })

        outputs = self.model(**model_inputs)

        start_logits = outputs.start_logits[0].detach().cpu()
        end_logits = outputs.end_logits[0].detach().cpu()
        gate_values = outputs.gate_values[0].detach().cpu()
        best_span = self._find_best_span(
            start_logits,
            end_logits,
            document_mask
        )
        if best_span is None:
            return {
                "answer": "",
                "bbox_normalized": [0, 0, 0, 0],
                "start_token": None,
                "end_token": None,
                "start_word": None,
                "end_word": None,
                "global_start_word": None,
                "global_end_word": None,
                "window_index": record.get("window_index"),
                "span_score": None,
                "gate_values": [],
            }

        start_token, end_token, span_score = best_span
        start_word = word_ids[start_token]
        end_word = word_ids[end_token]
        window_start = int(record.get("window_start", 0))
        return {
            "answer": " ".join(words[start_word:end_word + 1]),
            "bbox_normalized": union_boxes(boxes[start_word:end_word + 1]),
            "start_token": start_token,
            "end_token": end_token,
            "start_word": start_word,
            "end_word": end_word,
            "global_start_word": window_start + start_word,
            "global_end_word": window_start + end_word,
            "window_index": record.get("window_index"),
            "span_score": span_score,
            "gate_values": gate_values[start_token:end_token + 1].tolist(),
        }
