from typing import Any


def prompt_model(document: dict[str, Any], n_queries: int) -> dict[str, Any]:
    """
    Prompt the model with the document.

    Args:
        document: The document to prompt the model with.
        n_queries: The number of queries to prompt the model with.
        
    Returns:
        The document with the prompt.
    """
    return document