from torch import nn

class QAHead(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, hidden_states):
        logits = self.classifier(hidden_states)
        return logits[..., 0], logits[..., 1]