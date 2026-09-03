from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class EarlyStopping:
    patience: int = 3
    min_delta: float = 0.0
    best_loss: Optional[float] = None
    counter: int = 0

    def update(self, validation_loss: float) -> Tuple[bool, bool]:
        if self.best_loss is None:
            self.best_loss = validation_loss
            self.counter = 0
            return True, False

        improved = validation_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1

        should_stop = self.counter >= self.patience
        return improved, should_stop