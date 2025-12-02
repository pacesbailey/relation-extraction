import re
from typing import Callable

import chromadb
from datasets import Column, Dataset, DatasetDict


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


def get_collections(
    dataset: DatasetDict,
    client: chromadb.Client,
) -> dict[Combination, chromadb.Collection]:
    """
    Create collections for the dataset.

    Args:
        dataset: The dataset to create collections for.
        client: The chroma client to use.

    Returns:
        A dictionary containing the collections by combination.
    """
    collection_names: dict[Combination, str] = map_collection_names(dataset)
    collections: dict[Combination, chromadb.Collection] = {}
    for combination, collection_name in collection_names.items():
        collection: chromadb.Collection = client.get_or_create_collection(name=collection_name)
        if collection.count() == 0:
            filter: Callable = (
                lambda x: x["relation"] == combination[0] and
                x["subj_type"] == combination[1] and
                x["obj_type"] == combination[2]
            )
            subset: Dataset = dataset["train"].filter(filter)
            ids: list[int] = list(subset["id"])
            docs: list[str] = list(subset["text"])
            for idx in range(0, len(ids), (max := 5461)):
                collection.add(ids=ids[idx:idx+max], documents=docs[idx:idx+max])

        collections[combination] = collection

    return collections


if __name__ == "__main__":
    raise NotImplementedError
