from smolagents import Tool
from typing import Any, Optional

class SimpleTool(Tool):
    name = "classify_topic"
    description = "This tool classifies whether the given Vietnamese text is related to the specified topic."
    inputs = {"text":{"type":"string","description":"The Vietnamese text to be classified."},"topic":{"type":"string","description":"The string representing the topic to be checked."}}
    output_type = "boolean"

    def forward(self, text: str, topic: str) -> bool:
        """
        This tool classifies whether the given Vietnamese text is related to the specified topic.

        Args:
            text: The Vietnamese text to be classified.
            topic: The string representing the topic to be checked.

        Returns:
            bool: True if the text is related to the topic; False otherwise.
        """
        from transformers import pipeline
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        classifier = pipeline(
            "zero-shot-classification",
            model="vicgalle/xlm-roberta-large-xnli-anli",
            device=device,
            trust_remote_code=True,
        )

        candidate_labels = [topic, f"không liên quan {topic}"]
        result = classifier(text, candidate_labels)
        predicted_label = result["labels"][0]

        return predicted_label == topic