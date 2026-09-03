import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing.token_alignment import find_token_span

class SPDocVQADataset(Dataset):
    def __init__(
        self,
        json_path: str,
        processor,
        max_length: int = 512,
        strict_alignment: bool = True
    ):
        self.json_path = Path(json_path).resolve()
        self.project_root = self.json_path.parents[2]
        self.processor = processor
        self.max_length = max_length
        self.strict_alignment = strict_alignment

        if not self.json_path.exists():
            raise FileNotFoundError(f"File dataset tidak ditemukan: {self.json_path}")
        with self.json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            self.records = data
        elif isinstance(data, dict) and "data" in data:
            self.records = data["data"]
        else:
            raise ValueError("Format JSON harus berupa list atau dictionary yang memiliki key 'data'.")

        if len(self.records) == 0:
            raise ValueError(f"Dataset kosong: {self.json_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _encode_record(
        self,
        record: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        image_path = Path(record["image_path"])

        if not image_path.is_absolute():
            image_path = self.project_root / image_path

        if not image_path.exists():
            raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")

        question = str(record["question"])
        words = record["words"]
        boxes = record["boxes"]

        answer_start_word = int(record["answer_start_word"])
        answer_end_word = int(record["answer_end_word"])
        if len(words) != len(boxes):
            raise ValueError(f"Jumlah words dan boxes berbeda untuk question_id={record.get('question_id')}: words={len(words)}, boxes={len(boxes)}")

        with Image.open(image_path) as original_image:
            image = original_image.convert("RGB")

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

        start_token, end_token = find_token_span(
            word_ids=word_ids,
            sequence_ids=sequence_ids,
            answer_start_word=answer_start_word,
            answer_end_word=answer_end_word
        )

        if start_token is None or end_token is None:
            message = (
                "Jawaban tidak berhasil dipetakan ke token. "
                f"question_id={record.get('question_id')}, "
                f"answer_start_word={answer_start_word}, "
                f"answer_end_word={answer_end_word}. "
                "Kemungkinan window masih terlalu panjang sehingga jawaban terpotong saat tokenisasi."
            )
            if self.strict_alignment:
                raise ValueError(message)

            start_token = 0
            end_token = 0

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
            if isinstance(value, torch.Tensor)
        }

        item["start_positions"] = torch.tensor(start_token, dtype=torch.long)
        item["end_positions"] = torch.tensor(end_token, dtype=torch.long)

        token_ids = encoding["input_ids"][0].tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(token_ids)

        metadata = {
            "id": record.get("id"),
            "question_id": record.get("question_id"),
            "question": question,
            "answer": record.get("matched_answer"),
            "answers": record.get("answers", []),
            "answer_text_from_words": " ".join(words[answer_start_word:answer_end_word + 1]),
            "image_path": str(image_path),
            "answer_start_word": answer_start_word,
            "answer_end_word": answer_end_word,
            "start_token": start_token,
            "end_token": end_token,
            "answer_tokens": tokens[start_token:end_token + 1],
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
