import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from safetensors.torch import load_file, save_file

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
        dropout=0.1,
        use_oca=True,
        use_mcs=True
    ):
        super().__init__()
        self.adapter_size = adapter_size
        self.confidence_size = confidence_size
        self.dropout_probability = dropout
        self.backbone = LayoutLMv3Backbone(model_path)
        hidden_size = self.backbone.hidden_size
        self.confidence_adapter = None
        self.use_oca = use_oca
        self.use_mcs = use_mcs
        self.qa_head = QAHead(hidden_size=hidden_size, dropout=dropout)

        if self.use_oca:
            self.confidence_adapter = ConfidenceAdapter(
                hidden_size=hidden_size,
                adapter_size=adapter_size,
                confidence_size=confidence_size,
                dropout=dropout
            )

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

        gate_values = hidden_states.new_zeros(document_mask.shape)

        if self.use_oca:
            hidden_states, gate_values = self.confidence_adapter(
                hidden_states=hidden_states,
                bbox=bbox,
                confidence_ids=confidence_ids,
                question_mask=question_mask,
                document_mask=document_mask
            )

        start_logits, end_logits = self.qa_head(hidden_states)
        invalid_positions = ~document_mask.bool()
        start_logits = start_logits.masked_fill(invalid_positions, -10000.0)
        end_logits = end_logits.masked_fill(invalid_positions, -10000.0)

        loss = None
        if self.use_mcs:
            candidate_values = (
                candidate_start_positions,
                candidate_end_positions,
                candidate_weights,
                candidate_mask
            )

            if all(value is not None for value in candidate_values):
                loss = multi_span_loss(
                    start_logits=start_logits,
                    end_logits=end_logits,
                    candidate_start_positions=candidate_start_positions,
                    candidate_end_positions=candidate_end_positions,
                    candidate_weights=candidate_weights,
                    candidate_mask=candidate_mask
                )
            elif any(value is not None for value in candidate_values):
                raise ValueError("All candidate tensors must be provided together.")
        elif start_positions is not None and end_positions is not None:
            start_loss = nn.functional.cross_entropy(start_logits, start_positions)
            end_loss = nn.functional.cross_entropy(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
        elif start_positions is not None or end_positions is not None:
            raise ValueError("Start and end positions must be provided together.")

        return LayoutLMv3QAOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            gate_values=gate_values,
        )

    def checkpoint_config(self):
        return {
            "hidden_size": self.backbone.hidden_size,
            "adapter_size": self.adapter_size,
            "confidence_size": self.confidence_size,
            "dropout": self.dropout_probability,
            "use_oca": self.use_oca,
            "use_mcs": self.use_mcs
        }

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output_dir / "backbone")
        state_dict = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in self.state_dict().items()
            if not name.startswith("backbone.")
        }
        save_file(state_dict, output_dir / "model.safetensors")
        with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as file:
            json.dump(self.checkpoint_config(), file, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, checkpoint_dir, map_location="cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "adapter_config.json"
        backbone_path = checkpoint_dir / "backbone"
        if not config_path.exists() or not backbone_path.exists():
            raise FileNotFoundError(f"Model checkpoint is incomplete: {checkpoint_dir}")

        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        model = cls(
            model_path=backbone_path,
            adapter_size=config["adapter_size"],
            confidence_size=config["confidence_size"],
            dropout=config["dropout"],
            use_oca=config.get("use_oca", True),
            use_mcs=config.get("use_mcs", True)
        )

        safe_state_path = checkpoint_dir / "model.safetensors"
        model_state_path = checkpoint_dir / "model_state.pt"
        if safe_state_path.exists():
            missing, unexpected = model.load_state_dict(
                load_file(safe_state_path, device="cpu"),
                strict=False
            )
            if unexpected or any(not name.startswith("backbone.") for name in missing):
                raise ValueError(f"Model weights are incomplete: {safe_state_path}")
        elif model_state_path.exists():
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