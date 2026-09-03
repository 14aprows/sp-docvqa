import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collator import LayoutLMv3QACollator
from src.data.dataset import SPDocVQADataset
from src.models.model import build_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke training LayoutLMv3 untuk SP-DocVQA."
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train_sample.json"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/processed/val_sample.json"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/layoutlmv3-base"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/checkpoints/smoke_training"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Jumlah optimizer update untuk smoke training."
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=5
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()

def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT / path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
    }

def inspect_dataset(dataset: SPDocVQADataset):
    sample = dataset.inspect_item(0)

    print("\n" + "=" * 60)
    print("INSPEKSI SATU SAMPEL")
    print("=" * 60)
    print("Question ID:", sample["question_id"])
    print("Question:", sample["question"])
    print("Ground truth:", sample["answer"])
    print("Jawaban dari OCR:", sample["answer_text_from_words"])
    print("Word span:", sample["answer_start_word"], "-", sample["answer_end_word"])
    print("Token span:", sample["start_token"], "-", sample["end_token"])
    print("Answer tokens:", sample["answer_tokens"])
    print("Image:", sample["image_path"])
    print("\nTensor shapes:")

    for key, shape in sample["tensor_shapes"].items():
        print(f"  {key:18s}: {shape}")

    print("=" * 60 + "\n")

@torch.no_grad()
def validate(
    model,
    data_loader,
    device,
    maximum_batches: int
) -> float:
    model.eval()

    total_loss = 0.0
    batch_count = 0

    for batch_index, batch in enumerate(data_loader):
        if batch_index >= maximum_batches:
            break

        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)

        total_loss += outputs.loss.item()
        batch_count += 1

    if batch_count == 0:
        raise ValueError("Tidak ada batch validation.")

    return total_loss / batch_count

def main():
    args = parse_args()

    set_seed(args.seed)

    train_path = resolve_path(args.train_data)
    val_path = resolve_path(args.val_data)
    output_dir = resolve_path(args.output_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Train data tidak ditemukan: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Validation data tidak ditemukan: {val_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Model:", args.model_name)
    print("Train data:", train_path)
    print("Validation data:", val_path)
    print("Output directory:", output_dir)

    if device.type == "cpu":
        print(
            "\nPERINGATAN: CUDA tidak ditemukan. "
            "Program tetap dapat berjalan, tetapi LayoutLMv3 akan cukup lambat di CPU.\n"
        )

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        apply_ocr=False
    )

    train_dataset = SPDocVQADataset(
        json_path=str(train_path),
        processor=processor,
        max_length=args.max_length,
        strict_alignment=True
    )
    val_dataset = SPDocVQADataset(
        json_path=str(val_path),
        processor=processor,
        max_length=args.max_length,
        strict_alignment=True
    )

    print("Jumlah train:", len(train_dataset))
    print("Jumlah validation:", len(val_dataset))

    inspect_dataset(train_dataset)

    collator = LayoutLMv3QACollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )

    model = build_model(args.model_name)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    initial_batch = next(iter(train_loader))
    initial_batch = move_batch_to_device(initial_batch, device)

    model.eval()

    with torch.no_grad():
        initial_outputs = model(**initial_batch)

    print("\nForward-pass test berhasil.")
    print("Initial loss:", initial_outputs.loss.item())
    print("Start logits shape:", tuple(initial_outputs.start_logits.shape))
    print("End logits shape:", tuple(initial_outputs.end_logits.shape))

    model.train()
    optimizer.zero_grad()

    optimizer_step = 0
    recent_losses = []
    progress = tqdm(
        total=args.max_steps,
        desc="Smoke training"
    )

    while optimizer_step < args.max_steps:
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)

            outputs = model(**batch)
            raw_loss = outputs.loss
            scaled_loss = raw_loss / args.gradient_accumulation_steps

            scaled_loss.backward()
            recent_losses.append(raw_loss.item())

            is_accumulation_boundary = (batch_index + 1) % args.gradient_accumulation_steps == 0
            is_last_batch = batch_index + 1 == len(train_loader)

            if is_accumulation_boundary or is_last_batch:
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                optimizer_step += 1
                average_recent_loss = sum(recent_losses) / len(recent_losses)
                recent_losses = []

                progress.set_postfix(loss=f"{average_recent_loss:.4f}")
                progress.update(1)

                if optimizer_step >= args.max_steps:
                    break

    progress.close()

    validation_loss = validate(
        model=model,
        data_loader=val_loader,
        device=device,
        maximum_batches=args.validation_batches
    )

    print("\nSmoke training selesai.")
    print("Validation loss    :", validation_loss)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    report = {
        "status": "success",
        "model_name": args.model_name,
        "device": str(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "optimizer_steps": optimizer_step,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "validation_loss": validation_loss,
        "checkpoint_directory": str(output_dir)
    }

    report_path = output_dir / "smoke_training_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("Checkpoint:", output_dir)
    print("Training report:", report_path)

if __name__ == "__main__":
    main()