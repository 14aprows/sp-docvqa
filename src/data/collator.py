from typing import Dict, List
import torch 

class LayoutLMv3QACollator:
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(features) == 0:
            raise ValueError("Batch tidak boleh kosong.")

        expected_keys = set(features[0].keys())
        for index, feature in enumerate(features):
            current_keys = set(feature.keys())
            if current_keys != expected_keys:
                raise ValueError(
                    f"Key sampel ke-{index} berbeda. "
                    f"Expected={expected_keys}, "
                    f"received={current_keys}"
                )
            
        batch = {}

        for key in expected_keys:
            batch[key] = torch.stack(
                [feature[key] for feature in features]
            )

        return batch