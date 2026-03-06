from typing import Any

import dspy
from omegaconf import DictConfig


class ExtractRelation(dspy.Signature):
    input_text: str = dspy.InputField(description="The unlabeled input text.")
    output_text: str = dspy.OutputField(description="The output text with labeled entities and relations.")


class RelationExtractor(dspy.Module):
    def __init__(self, prompt: str) -> None:
        """
        Initialize the relation extractor module with a given prompt.

        Args:
            prompt: The prompt to use for the relation extractor.
        """
        super().__init__()
        signature: dspy.Signature = ExtractRelation.with_instructions(prompt)
        self.extractor: dspy.ChainOfThought = dspy.ChainOfThought(signature)

    def forward(self, input_text: str) -> str:
        """
        Forward pass for the relation extractor.

        Args:
            input_text: The unlabeled input text.

        Returns:
            The output text with labeled entities and relations.
        """
        return self.extractor(input_text=input_text).output_text


def configure_prompt(config: DictConfig, examples: dict, document: dict) -> str:
    """
    Configure the prompt for the relation extractor.

    Args:
        config: The configuration for the relation extractor.
        examples: The examples to use for the relation extractor.
        document: The document to use for the relation extractor.

    Returns:
        The configured prompt.
    """
    task: str = config.prompt.task.format(
        relation_type=document["relation"],
        head_type=document["subj_type"],
        tail_type=document["obj_type"],
        description=config.dataset.label_types[document["relation"]]["description"],
    )
    prompt: str = f"{config.prompt.system} {task}"
    for input, output in zip(examples["documents"][0], examples["metadatas"][0]):
        prompt += config.prompt.example.format(input=input, output=output["labeled_text"])

    return prompt + config.prompt.user.format(input=document["text"])



if __name__ == "__main__":
    raise NotImplementedError
