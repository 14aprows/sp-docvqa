from typing import Dict, List
import torch 

class LayoutLMv3QACollator:
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(features) == 0:
            raise ValueError("Batch cannot be empty.")

        expected_keys = set(features[0].keys())
        for index, feature in enumerate(features):
            current_keys = set(feature.keys())
            if current_keys != expected_keys:
                raise ValueError(
                    f"Sample {index} has different keys. "
                    f"Expected={expected_keys}, "
                    f"received={current_keys}"
                )
            
        batch = {}

        for key in expected_keys:
            batch[key] = torch.stack(
                [feature[key] for feature in features]
            )

        return batch