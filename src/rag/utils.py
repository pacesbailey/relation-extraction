"""Utils module for the RAG pipeline.

This module contains utility functions for the RAG pipeline.

Example:
>>> dataset: DatasetDict = load_dataset("json", data_dir="data")
>>> collection_names: dict[Combination, str] = map_collection_names(dataset)
"""

import re
from copy import deepcopy
from typing import Any

from datasets import DatasetDict, Column

type Combination = tuple[str, str, str]


def clean_string(text: str, replacement: str = '_') -> str:
    """
    Clean a string by replacing all non-alphanumeric characters with a
    replacement character.

    Args:
        text: The string to clean.
        replacement: The replacement character.

    Returns:
        The cleaned string.
    """
    return re.sub(r'[^a-zA-Z0-9]', replacement, text)


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
        "token": document_copy["token"],
        "relation": document_copy["relation"],
        "subj_type": document_copy["subj_type"],
        "subj_start": document_copy["subj_start"],
        "subj_end": document_copy["subj_end"],
        "obj_type": document_copy["obj_type"],
        "obj_start": document_copy["obj_start"],
        "obj_end": document_copy["obj_end"],
        "stanford_ner": document_copy["stanford_ner"],
    }


def map_collection_names(dataset: DatasetDict) -> dict[Combination, str]:
    """
    Maps unique relation, subj_type, and obj_type combinations to
    chromadb-friendly collection names.

    Args:
        dataset: The dataset to map collection names for.

    Returns:
        A dictionary containing the collection names by combination.
    """
    relations: Column = dataset["train"]["relation"]
    subj_types: Column = dataset["train"]["subj_type"]
    obj_types: Column = dataset["train"]["obj_type"]
    unique_combinations: set[Combination] = set(zip(relations, subj_types, obj_types))
    
    return {
        combination: clean_string("-".join(combination))
        for combination in unique_combinations
    }