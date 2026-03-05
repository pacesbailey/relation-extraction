"""Utils module for the RAG pipeline.

This module contains utility functions for the RAG pipeline.

Example:
>>> dataset: DatasetDict = load_dataset("json", data_dir="data")
>>> collection_names: dict[Combination, str] = map_collection_names(dataset)
"""

import re
from copy import deepcopy
from typing import Any


def clean_string(text: str, replacement: str = "_") -> str:
    """
    Clean a string by replacing all non-alphanumeric characters with a
    replacement character.

    Args:
        text: The string to clean.
        replacement: The replacement character.

    Returns:
        The cleaned string.
    """
    return re.sub(r"[^a-zA-Z0-9]", replacement, text)


def get_metadata(document: dict[str, Any]) -> dict[str, Any]:
    """
    Get the metadata for a document.

    Args:
        document: The document to get metadata for.

    Returns:
        The metadata for the document.
    """
    document_copy: dict = deepcopy(document)

    return {
        "relation": document_copy["relation"],
        "subj_type": document_copy["subj_type"],
        "obj_type": document_copy["obj_type"],
    }


if __name__ == "__main__":
    raise NotImplementedError
