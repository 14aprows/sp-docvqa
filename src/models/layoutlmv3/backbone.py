from torch import nn
from transformers import LayoutLMv3Model

class LayoutLMv3Backbone(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.layoutlmv3 = LayoutLMv3Model.from_pretrained(model_path)
        self.hidden_size = self.layoutlmv3.config.hidden_size

    def forward(
        self,
        input_ids,
        bbox,
        pixel_values,
        attention_mask,
        token_type_ids=None
    ):
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        text_length = input_ids.size(1)
        return outputs.last_hidden_state[:, :text_length, :]

    def save_pretrained(self, output_dir):
        self.layoutlmv3.save_pretrained(output_dir)