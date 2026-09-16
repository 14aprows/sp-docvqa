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

def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path

def load_config(path_value):
    path = resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Config tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config tidak valid: {path}")
    return config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_optimizer(model, training_config):
    parameter_groups = (
        {
            "params": model.backbone.parameters(),
            "lr": training_config["backbone_learning_rate"],
        },
        {
            "params": list(model.confidence_adapter.parameters())
            + list(model.qa_head.parameters()),
            "lr": training_config["new_layers_learning_rate"],
        },
    )
    return AdamW(
        parameter_groups,
        weight_decay=training_config["weight_decay"],
    )

def main():
    config = load_config("configs/layoutlmv3.yaml")
    model_config = config["model"]
    data_config = config["data"]
    preprocessing_config = config["preprocessing"]
    training_config = config["training"]

    set_seed(training_config["seed"])
    train_path = resolve_path(data_config["train_path"])
    val_path = resolve_path(data_config["val_path"])
    output_dir = resolve_path(training_config["output_dir"])
    resume_value = training_config.get("resume_from")
    resume_path = resolve_path(resume_value) if resume_value else None
    model_source = resume_path or model_config["pretrained_model"]

    for path in (train_path, val_path):
        if not path.exists():
            raise FileNotFoundError(f"Input training tidak ditemukan: {path}")
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {resume_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        model_source,
        apply_ocr=False,
    )
    dataset_arguments = {
        "processor": processor,
        "max_length": model_config["max_length"],
        "max_candidates": preprocessing_config["max_candidates"],
        "require_candidates": True,
    }
    train_dataset = SPDocVQADataset(
        json_path=str(train_path),
        **dataset_arguments,
    )
    val_dataset = SPDocVQADataset(
        json_path=str(val_path),
        **dataset_arguments,
    )

    collator = LayoutLMv3QACollator()
    num_workers = training_config["num_workers"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["train_batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["val_batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = build_model(
        model_path=model_source,
        adapter_size=model_config["adapter_size"],
        confidence_size=model_config["confidence_size"],
        dropout=model_config["dropout"],
        map_location=device,
    )
    optimizer = create_optimizer(model, training_config)
    accumulation_steps = training_config["gradient_accumulation_steps"]
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = updates_per_epoch * training_config["epochs"]
    warmup_steps = int(total_steps * training_config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("Device:", device)
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Model source:", model_source)

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
        gradient_accumulation_steps=accumulation_steps,
        max_grad_norm=training_config["max_grad_norm"],
        mixed_precision=training_config["mixed_precision"],
        early_stopping_patience=training_config["early_stopping_patience"],
        early_stopping_min_delta=training_config["early_stopping_min_delta"],
    )
    if resume_path:
        trainer.load_training_state(str(resume_path))
    trainer.fit()

if __name__ == "__main__":
    main()