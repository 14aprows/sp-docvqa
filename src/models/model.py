from pathlib import Path
from src.models.layoutlmv3.model import LayoutLMv3ForAnswerLocalization

def build_model(
    model_path,
    adapter_size=256,
    confidence_size=32,
    dropout=0.1,
    map_location="cpu"
):
    source_path = Path(model_path)
    if source_path.exists() and (source_path / "adapter_config.json").exists():
        return LayoutLMv3ForAnswerLocalization.from_pretrained(
            source_path,
            map_location=map_location
        )

    return LayoutLMv3ForAnswerLocalization(
        model_path=model_path,
        adapter_size=adapter_size,
        confidence_size=confidence_size,
        dropout=dropout
    )