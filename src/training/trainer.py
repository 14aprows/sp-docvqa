import json
from pathlib import Path
from typing import Dict

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.training.callbacks import EarlyStopping

class LayoutLMv3Trainer:
    def __init__(
        self,
        model,
        processor,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device: torch.device,
        output_dir: str,
        epochs: int,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = True,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 0.0
    ):
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device
        self.output_dir = Path(output_dir)
        self.epochs = epochs

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience, min_delta=early_stopping_min_delta
        )
        self.history = []
        self.start_epoch = 1
        self.global_step = 0

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def _move_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(self.device)
            for key, value in batch.items()
        }

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        batch_count = 0

        progress = tqdm(self.train_loader, desc=f"Train epoch {epoch}")
        for batch_index, batch in enumerate(progress):
            batch = self._move_batch(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp
            ):
                outputs = self.model(**batch)
                raw_loss = outputs.loss
                scaled_loss = raw_loss / self.gradient_accumulation_steps

            self.scaler.scale(scaled_loss).backward()

            total_loss += raw_loss.item()
            batch_count += 1

            accumulation_finished = (batch_index + 1) % self.gradient_accumulation_steps == 0
            final_batch = (batch_index + 1) == len(self.train_loader)
            if accumulation_finished or final_batch:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            progress.set_postfix(
                loss=f"{raw_loss.item():.4f}",
                step=self.global_step
            )

        if batch_count == 0:
            raise ValueError("Train DataLoader kosong.")

        return total_loss / batch_count

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()

        total_loss = 0.0
        batch_count = 0

        progress = tqdm(self.val_loader, desc=f"Validation epoch {epoch}")

        for batch in progress:
            batch = self._move_batch(batch)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp
            ):
                outputs = self.model(**batch)
                loss = outputs.loss

            total_loss += loss.item()
            batch_count += 1

            progress.set_postfix(loss=f"{loss.item():.4f}")

        if batch_count == 0:
            raise ValueError("Validation DataLoader kosong.")

        return total_loss / batch_count

    def save_checkpoint(
        self,
        checkpoint_name: str,
        epoch: int,
        train_loss: float,
        validation_loss: float
    ):
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)

        training_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "history": self.history,
            "early_stopping_best_loss": self.early_stopping.best_loss,
            "early_stopping_counter": self.early_stopping.counter
        }

        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict(),
                "training_state": training_state
            },
            checkpoint_dir / "training_state.pt"
        )

        with (checkpoint_dir / "training_history.json").open("w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2, ensure_ascii=False)

    def load_training_state(self, checkpoint_dir: str):
        state_path = Path(checkpoint_dir) / "training_state.pt"

        if not state_path.exists():
            raise FileNotFoundError(f"Training state tidak ditemukan: {state_path}")

        checkpoint = torch.load(state_path, map_location=self.device)

        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])

        training_state = checkpoint["training_state"]

        self.global_step = training_state["global_step"]
        self.history = training_state.get("history", [])
        self.start_epoch = training_state["epoch"] + 1
        self.early_stopping.best_loss = training_state.get(
            "early_stopping_best_loss",
            training_state.get("validation_loss")
        )
        self.early_stopping.counter = training_state.get("early_stopping_counter", 0)

    def fit(self):
        self.model.to(self.device)

        print("\nTraining dimulai")
        print("Device:", self.device)
        print("Start epoch:", self.start_epoch)
        print("Total epoch:", self.epochs)
        print("AMP aktif:", self.use_amp)

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            validation_loss = self.validate(epoch)

            epoch_result = {
                "epoch": epoch,
                "global_step": self.global_step,
                "train_loss": train_loss,
                "validation_loss": validation_loss
            }
            self.history.append(epoch_result)

            improved, should_stop = self.early_stopping.update(validation_loss)

            self.save_checkpoint(
                checkpoint_name="last",
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss
            )

            if improved:
                self.save_checkpoint(
                    checkpoint_name="best",
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss
                )

            print("\n" + "=" * 60)
            print("Epoch:", epoch)
            print("Train loss:", train_loss)
            print("Validation loss:", validation_loss)
            print("Best loss:", self.early_stopping.best_loss)
            print("Patience counter:", self.early_stopping.counter)
            print("=" * 60)

            if should_stop:
                print("\nEarly stopping aktif. Training dihentikan.")
                break

        print("\nTraining selesai.")
        print("Best checkpoint:", self.output_dir / "best")
        print("Last checkpoint:", self.output_dir / "last")