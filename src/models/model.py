from transformers import LayoutLMv3ForQuestionAnswering

def build_model(model_name: str = "microsoft/layoutlmv3-base"):
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_name)
    return model