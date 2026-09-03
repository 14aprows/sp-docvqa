from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

from src.evaluation.iou import union_boxes

class LayoutLMv3QAPredictor:
    def __init__(
        self,
        model,
        processor,
        device: torch.device,
        max_length: int = 512,
        max_answer_tokens: int = 30,
        top_k: int = 20
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.max_length = max_length
        self.max_answer_tokens = max_answer_tokens
        self.top_k = top_k

        self.model.to(device)
        self.model.eval()

    def _find_best_span(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        word_ids: List,
        sequence_ids: List
    ):
        valid_positions = [
            token_index
            for token_index, (word_id, sequence_id) in enumerate(zip(word_ids, sequence_ids))
            if sequence_id == 1 and word_id is not None
        ]

        if not valid_positions:
            return None

        masked_start = torch.full_like(start_logits, float("-inf"))
        masked_end = torch.full_like(end_logits, float("-inf"))
        masked_start[valid_positions] = start_logits[valid_positions]
        masked_end[valid_positions] = end_logits[valid_positions]

        candidate_count = min(self.top_k, len(valid_positions))
        start_candidates = torch.topk(masked_start, k=candidate_count).indices.tolist()
        end_candidates = torch.topk(masked_end, k=candidate_count).indices.tolist()

        best_span = None
        best_score = float("-inf")

        for start_token in start_candidates:
            for end_token in end_candidates:
                span_length = end_token - start_token + 1

                if end_token < start_token or span_length > self.max_answer_tokens:
                    continue

                score = start_logits[start_token].item() + end_logits[end_token].item()

                if score > best_score:
                    best_score = score
                    best_span = (start_token, end_token)

        return best_span

    @torch.no_grad()
    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        image_path = Path(record["image_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

        words = record["words"]
        boxes = record["boxes"]
        question = record["question"]

        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        encoding = self.processor(
            image,
            question,
            words,
            boxes=boxes,
            truncation="only_second",
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        sequence_ids = encoding.sequence_ids(batch_index=0)

        model_inputs = {
            key: value.to(self.device)
            for key, value in encoding.items()
            if isinstance(value, torch.Tensor)
        }

        outputs = self.model(**model_inputs)

        start_logits = outputs.start_logits[0].detach().cpu()
        end_logits = outputs.end_logits[0].detach().cpu()

        best_span = self._find_best_span(
            start_logits=start_logits,
            end_logits=end_logits,
            word_ids=word_ids,
            sequence_ids=sequence_ids
        )

        if best_span is None:
            return {
                "answer": "",
                "bbox": [0, 0, 0, 0],
                "start_token": None,
                "end_token": None,
                "start_word": None,
                "end_word": None,
                "span_score": None
            }

        start_token, end_token = best_span
        start_word = word_ids[start_token]
        end_word = word_ids[end_token]

        answer_words = words[start_word:end_word + 1]
        answer_boxes = boxes[start_word:end_word + 1]

        predicted_answer = " ".join(answer_words)
        predicted_bbox = union_boxes(answer_boxes)
        span_score = start_logits[start_token].item() + end_logits[end_token].item()

        return {
            "answer": predicted_answer,
            "bbox": predicted_bbox,
            "start_token": start_token,
            "end_token": end_token,
            "start_word": start_word,
            "end_word": end_word,
            "span_score": span_score
        }
