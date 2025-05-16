from smolagents import Tool
from typing import Any, Optional

class SimpleTool(Tool):
    name = "summarize_news"
    description = "This tool summarizes the given Vietnamese news text."
    inputs = {"text":{"type":"string","description":"The Vietnamese news text to be summarized."}}
    output_type = "string"

    def forward(self, text: str) -> str:
        """
        This tool summarizes the given Vietnamese news text.

        Args:
            text (str): The Vietnamese news text to be summarized.

        Returns:
            str: The summarized version of the input text.
        """
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "VietAI/vit5-base-vietnews-summarization"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()

        formatted_text = "vietnews: " + text + " </s>"
        encoding = tokenizer(formatted_text, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_masks = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_masks,max_length=256)

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return summary