import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collator import LayoutLMv3QACollator
from src.data.dataset import SPDocVQADataset
from src.models.model import build_model
from src.training.trainer import LayoutLMv3Trainer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full training LayoutLMv3 untuk SP-DocVQA."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml"
    )

    return parser.parse_args()

def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT / path

def load_config(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"Config tidak ditemukan: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config tidak valid: {config_path}")

    return config

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()

    config_path = resolve_path(args.config)
    config = load_config(config_path)

    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]

    set_seed(training_config["seed"])

    train_path = resolve_path(data_config["train_path"])
    val_path = resolve_path(data_config["val_path"])
    output_dir = resolve_path(training_config["output_dir"])

    if not train_path.exists():
        raise FileNotFoundError(f"Train data tidak ditemukan: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Validation data tidak ditemukan: {val_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_from = training_config.get("resume_from")
    resume_path = resolve_path(resume_from) if resume_from else None
    model_source = str(resume_path) if resume_path else model_config["name"]

    if resume_path and not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {resume_path}")

    print("Device:", device)
    print("Model source:", model_source)
    print("Train data:", train_path)
    print("Validation data:", val_path)
    print("Output directory:", output_dir)

    if device.type == "cpu":
        print(
            "\nPERINGATAN: CUDA tidak ditemukan. "
            "Full training LayoutLMv3 akan sangat lambat di CPU.\n"
        )

    processor = AutoProcessor.from_pretrained(
        model_source,
        apply_ocr=False
    )

    train_dataset = SPDocVQADataset(
        json_path=str(train_path),
        processor=processor,
        max_length=model_config["max_length"],
        strict_alignment=True
    )
    val_dataset = SPDocVQADataset(
        json_path=str(val_path),
        processor=processor,
        max_length=model_config["max_length"],
        strict_alignment=True
    )

    print("Jumlah train:", len(train_dataset))
    print("Jumlah validation:", len(val_dataset))

    collator = LayoutLMv3QACollator()
    num_workers = training_config["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["train_batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["val_batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0
    )

    model = build_model(model_source)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"]
    )

    gradient_accumulation_steps = training_config["gradient_accumulation_steps"]
    updates_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_training_steps = updates_per_epoch * training_config["epochs"]
    warmup_steps = int(total_training_steps * training_config["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    trainer = LayoutLMv3Trainer(
        model=model,
        processor=processor,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=str(output_dir),
        epochs=training_config["epochs"],
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=training_config["max_grad_norm"],
        mixed_precision=training_config["mixed_precision"],
        early_stopping_patience=training_config["early_stopping_patience"],
        early_stopping_min_delta=training_config["early_stopping_min_delta"]
    )

    if resume_path:
        trainer.load_training_state(str(resume_path))

    trainer.fit()

if __name__ == "__main__":
    main()