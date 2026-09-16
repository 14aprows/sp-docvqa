import torch
from torch import nn

class ConfidenceAdapter(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        adapter_size=256,
        confidence_size=32,
        dropout=0.1
    ):
        super().__init__()
        self.confidence_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=confidence_size,
            padding_idx=0
        )
        self.token_projection = nn.Linear(hidden_size, adapter_size)
        self.question_projection = nn.Linear(hidden_size, adapter_size)
        self.spatial_projection = nn.Sequential(
            nn.Linear(6, adapter_size),
            nn.GELU()
        )

        combined_size = adapter_size * 3 + confidence_size
        self.feature_projection = nn.Sequential(
            nn.Linear(combined_size, adapter_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(
            nn.Linear(combined_size, 1),
            nn.Sigmoid()
        )
        self.output_projection = nn.Linear(adapter_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def pool_question(self, hidden_states, question_mask):
        mask = question_mask.unsqueeze(-1).to(hidden_states.dtype)
        question_sum = (hidden_states * mask).sum(dim=1)
        question_length = mask.sum(dim=1).clamp_min(1.0)
        return self.question_projection(question_sum / question_length)

    def create_spatial_features(self, bbox):
        bbox = bbox.to(dtype=torch.float32) / 1000.0
        x1, y1, x2, y2 = bbox.unbind(dim=-1)
        width = (x2 - x1).clamp_min(0.0)
        height = (y2 - y1).clamp_min(0.0)
        spatial_values = torch.stack(
            (x1, y1, x2, y2, width, height), dim=-1
        )
        return self.spatial_projection(spatial_values)

    def forward(
        self,
        hidden_states,
        bbox,
        confidence_ids,
        question_mask,
        document_mask
    ):
        token_features = self.token_projection(hidden_states)
        question_features = self.pool_question(hidden_states, question_mask)
        question_features = question_features.unsqueeze(1).expand(
            -1, hidden_states.size(1), -1
        )
        confidence_features = self.confidence_embedding(confidence_ids.long())
        confidence_features = confidence_features.to(token_features.dtype)
        spatial_features = self.create_spatial_features(bbox).to(token_features.dtype)
        combined_features = torch.cat((
            token_features,
            question_features,
            confidence_features,
            spatial_features
        ), dim=-1)

        adapter_features = self.feature_projection(combined_features)
        gate_values = self.gate(combined_features)
        update = self.output_projection(gate_values * adapter_features)
        document_mask_3d = document_mask.unsqueeze(-1).bool()
        update = update * document_mask_3d.to(update.dtype)

        adapted_states = self.output_norm(hidden_states + self.dropout(update))
        final_hidden_states = torch.where(
            document_mask_3d,
            adapted_states,
            hidden_states
        )
        masked_gate_values = gate_values.squeeze(-1) * document_mask.to(gate_values.dtype)
        
        return final_hidden_states, masked_gate_values