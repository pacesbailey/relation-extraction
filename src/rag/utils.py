"""Utils module for the RAG pipeline.

This module contains utility functions for the RAG pipeline.

Example:
>>> dataset: DatasetDict = load_dataset("json", data_dir="data")
>>> collection_names: dict[Combination, str] = map_collection_names(dataset)
"""

import re
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