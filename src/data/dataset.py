import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing.token_alignment import find_token_span

def is_low_confidence_label(label):
    return label is not None and str(label).strip().lower() == "low"

def build_token_level_features(
    word_ids,
    sequence_ids,
    confidence_labels,
    attention_mask
):
    if len(word_ids) != len(sequence_ids):
        raise ValueError("Jumlah word_ids dan sequence_ids harus sama")

    confidence_ids = torch.zeros(len(word_ids), dtype=torch.long)
    question_mask = torch.zeros(len(word_ids), dtype=torch.long)
    document_mask = torch.zeros(len(word_ids), dtype=torch.long)

    for token_index, (word_id, sequence_id) in enumerate(zip(word_ids, sequence_ids)):
        if int(attention_mask[token_index]) == 0:
            continue
        if sequence_id == 0:
            question_mask[token_index] = 1
            continue
        if sequence_id != 1 or word_id is None:
            continue
        if not 0 <= word_id < len(confidence_labels):
            raise ValueError(f"word_id di luar confidence_labels: {word_id}")

        document_mask[token_index] = 1
        confidence_ids[token_index] = 2 if is_low_confidence_label(confidence_labels[word_id]) else 1

    return confidence_ids, question_mask, document_mask

class SPDocVQADataset(Dataset):
    def __init__(
        self,
        json_path: str,
        processor,
        max_length: int = 512,
        max_candidates: int = None,
        require_candidates: bool = True
    ):
        self.json_path = Path(json_path).resolve()
        self.project_root = self.json_path.parents[2]
        self.processor = processor
        self.max_length = max_length
        self.require_candidates = require_candidates

        if not self.json_path.exists():
            raise FileNotFoundError(f"File dataset tidak ditemukan: {self.json_path}")
        
        with self.json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if isinstance(payload, list):
            self.records = payload
            payload_max_candidates = 20
        elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
            self.records = payload["data"]
            payload_max_candidates = int(payload.get("max_candidates", 20))
        else:
            raise ValueError("Format JSON harus berupa list atau dictionary yang memiliki key 'data'.")

        if not self.records:
            raise ValueError(f"Dataset kosong: {self.json_path}")

        self.max_candidates = max_candidates or payload_max_candidates
        if self.max_candidates <= 0:
            raise ValueError("max_candidates harus lebih besar dari 0")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, image_path):
        path = Path(image_path)
        return path if path.is_absolute() else self.project_root / path

    def _encode_candidates(self, record, word_ids, sequence_ids):
        starts = torch.zeros(self.max_candidates, dtype=torch.long)
        ends = torch.zeros(self.max_candidates, dtype=torch.long)
        weights = torch.zeros(self.max_candidates, dtype=torch.float)
        mask = torch.zeros(self.max_candidates, dtype=torch.bool)

        valid_candidates = []
        for candidate in record.get("candidate_spans", []):
            start_token, end_token = find_token_span(
                word_ids=word_ids,
                sequence_ids=sequence_ids,
                answer_start_word=int(candidate["start_word"]),
                answer_end_word=int(candidate["end_word"])
            )
            if start_token is None or end_token is None:
                continue
            if len(valid_candidates) >= self.max_candidates:
                break

            candidate_index = len(valid_candidates)
            starts[candidate_index] = start_token
            ends[candidate_index] = end_token
            weights[candidate_index] = float(candidate["weight"])
            mask[candidate_index] = True
            valid_candidates.append({
                **candidate,
                "start_token": start_token,
                "end_token": end_token
            })

        if self.require_candidates and not valid_candidates:
            record_id = record.get("id")
            question_id = record.get("question_id")
            raise ValueError(f"Sample tidak memiliki kandidat token valid: id={record_id}, question_id={question_id}")
        if valid_candidates and weights[mask].sum().item() <= 0:
            record_id = record.get("id")
            raise ValueError(f"Bobot kandidat harus positif: id={record_id}")

        return starts, ends, weights, mask, valid_candidates

    def _encode_record(
        self,
        record: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        image_path = self._resolve_image_path(record["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")

        question = str(record["question"])
        words = record["words"]
        boxes = record["boxes"]
        confidence_labels = record["confidence_labels"]
        if not len(words) == len(boxes) == len(confidence_labels):
            record_id = record.get("id")
            raise ValueError(f"Panjang words, boxes, dan confidence_labels berbeda: id={record_id}")

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
        attention_mask = encoding["attention_mask"][0]

        confidence_ids, question_mask, document_mask = build_token_level_features(
            word_ids=word_ids,
            sequence_ids=sequence_ids,
            confidence_labels=confidence_labels,
            attention_mask=attention_mask
        )

        (
            candidate_start_positions,
            candidate_end_positions,
            candidate_weights,
            candidate_mask,
            valid_candidates
        ) = self._encode_candidates(
            record,
            word_ids,
            sequence_ids
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
            if isinstance(value, torch.Tensor)
        }
        item.update({
            "confidence_ids": confidence_ids,
            "question_mask": question_mask,
            "document_mask": document_mask,
            "candidate_start_positions": candidate_start_positions,
            "candidate_end_positions": candidate_end_positions,
            "candidate_weights": candidate_weights,
            "candidate_mask": candidate_mask
        })

        if valid_candidates:
            item["start_positions"] = candidate_start_positions[0].clone()
            item["end_positions"] = candidate_end_positions[0].clone()

        token_ids = encoding["input_ids"][0].tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(token_ids)
        metadata = {
            "id": record.get("id"),
            "question_id": record.get("question_id"),
            "question": question,
            "answers": record.get("answers", []),
            "image_path": str(image_path),
            "valid_candidates": valid_candidates,
            "candidate_count": len(valid_candidates),
            "tokens": tokens,
            "word_ids": word_ids,
            "sequence_ids": sequence_ids
        }

        return item, metadata

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        item, _ = self._encode_record(record)

        return item

    def inspect_item(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        item, metadata = self._encode_record(record)

        metadata["tensor_shapes"] = {
            key: list(value.shape)
            for key, value in item.items()
        }

        return metadata