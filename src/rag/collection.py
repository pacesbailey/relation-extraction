"""Collection module for the relation extraction task.

This module contains functions for creating and managing collections of
documents for the relation extraction task. The collections are created
using the chromadb library and are stored in a persistent client.

Example:
>>> dataset: DatasetDict = load_dataset("json", data_dir="data")
>>> client: chromadb.Client = chromadb.PersistentClient(path="chroma")
>>> collections: dict[Combination, chromadb.Collection] = get_collections(dataset, client)
>>> collection: chromadb.Collection = collections[("relation", "subj_type", "obj_type")]
>>> results: dict = collection.query(query_texts=["query text"], n_results=4)
"""

from typing import Callable

import chromadb
from datasets import Dataset, DatasetDict

from .utils import Combination, map_collection_names


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
