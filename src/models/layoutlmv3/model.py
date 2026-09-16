import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from src.models.layoutlmv3.backbone import LayoutLMv3Backbone
from src.models.layoutlmv3.confidence_adapter import ConfidenceAdapter
from src.models.layoutlmv3.multi_span_loss import multi_span_loss
from src.models.layoutlmv3.qa_head import QAHead

@dataclass
class LayoutLMv3QAOutput:
    loss: Optional[torch.Tensor]
    start_logits: torch.Tensor
    end_logits: torch.Tensor
    gate_values: torch.Tensor

class LayoutLMv3ForAnswerLocalization(nn.Module):
    def __init__(
        self,
        model_path,
        adapter_size=256,
        confidence_size=32,
        dropout=0.1
    ):
        super().__init__()
        self.adapter_size = adapter_size
        self.confidence_size = confidence_size
        self.dropout_probability = dropout
        self.backbone = LayoutLMv3Backbone(model_path)
        hidden_size = self.backbone.hidden_size
        self.confidence_adapter = ConfidenceAdapter(
            hidden_size=hidden_size,
            adapter_size=adapter_size,
            confidence_size=confidence_size,
            dropout=dropout
        )
        self.qa_head = QAHead(hidden_size=hidden_size, dropout=dropout)

    def forward(
        self,
        input_ids,
        bbox,
        pixel_values,
        attention_mask,
        confidence_ids,
        question_mask,
        document_mask,
        candidate_start_positions=None,
        candidate_end_positions=None,
        candidate_weights=None,
        candidate_mask=None,
        start_positions=None,
        end_positions=None,
        token_type_ids=None
    ):
        hidden_states = self.backbone(
            input_ids=input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        adapted_hidden_states, gate_values = self.confidence_adapter(
            hidden_states=hidden_states,
            bbox=bbox,
            confidence_ids=confidence_ids,
            question_mask=question_mask,
            document_mask=document_mask
        )
        start_logits, end_logits = self.qa_head(adapted_hidden_states)

        invalid_positions = ~document_mask.bool()
        start_logits = start_logits.masked_fill(invalid_positions, -10000.0)
        end_logits = end_logits.masked_fill(invalid_positions, -10000.0)

        candidate_values = (
            candidate_start_positions,
            candidate_end_positions,
            candidate_weights,
            candidate_mask
        )
        provided_candidate_values = [value is not None for value in candidate_values]
        if any(provided_candidate_values) and not all(provided_candidate_values):
            raise ValueError("Seluruh tensor candidate_* harus diberikan bersama sama")

        loss = None
        if all(provided_candidate_values):
            loss = multi_span_loss(
                start_logits=start_logits,
                end_logits=end_logits,
                candidate_start_positions=candidate_start_positions,
                candidate_end_positions=candidate_end_positions,
                candidate_weights=candidate_weights,
                candidate_mask=candidate_mask
            )

        return LayoutLMv3QAOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            gate_values=gate_values
        )

    def checkpoint_config(self):
        return {
            "hidden_size": self.backbone.hidden_size,
            "adapter_size": self.adapter_size,
            "confidence_size": self.confidence_size,
            "dropout": self.dropout_probability
        }

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output_dir / "backbone")
        torch.save(
            self.confidence_adapter.state_dict(),
            output_dir / "confidence_adapter.pt"
        )
        torch.save(self.qa_head.state_dict(), output_dir / "qa_head.pt")
        torch.save(self.state_dict(), output_dir / "model_state.pt")
        with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as file:
            json.dump(self.checkpoint_config(), file, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, checkpoint_dir, map_location="cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "adapter_config.json"
        backbone_path = checkpoint_dir / "backbone"
        if not config_path.exists() or not backbone_path.exists():
            raise FileNotFoundError(f"Checkpoint model final tidak lengkap: {checkpoint_dir}")

        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        model = cls(
            model_path=backbone_path,
            adapter_size=config["adapter_size"],
            confidence_size=config["confidence_size"],
            dropout=config["dropout"]
        )

        model_state_path = checkpoint_dir / "model_state.pt"
        if model_state_path.exists():
            state_dict = torch.load(
                model_state_path,
                map_location=map_location,
                weights_only=True
            )
            model.load_state_dict(state_dict)
        else:
            adapter_state = torch.load(
                checkpoint_dir / "confidence_adapter.pt",
                map_location=map_location,
                weights_only=True
            )
            qa_head_state = torch.load(
                checkpoint_dir / "qa_head.pt",
                map_location=map_location,
                weights_only=True
            )
            model.confidence_adapter.load_state_dict(adapter_state)
            model.qa_head.load_state_dict(qa_head_state)

        return model