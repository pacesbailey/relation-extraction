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

from typing import Any, Callable

import chromadb
from datasets import Dataset, DatasetDict

from .utils import Combination, get_metadata, map_collection_names


def add_documents(
    collection: chromadb.Collection,
    dataset: Dataset,
    relation: str,
    subj_type: str,
    obj_type: str,
    batch_size: int = 5461,
) -> None:
    """
    Adds documents to a collection based on a given relation, subj_type, and 
    obj_type.

    Args:
        collection: The collection to add documents to.
        dataset: The dataset to add documents from.
        relation: The relation to add documents for.
        subj_type: The subj_type to add documents for.
        obj_type: The obj_type to add documents for.
        batch_size: The number of documents to add in each batch.
    """
    filter_func: Callable = (
        lambda x: x["relation"] == relation and
        x["subj_type"] == subj_type and
        x["obj_type"] == obj_type
    )
    subset: Dataset = dataset.filter(filter_func)
    ids: list[int] = list(subset["id"])
    documents: list[str] = list(subset["text"])
    metadata: list[dict[str, Any]] = [get_metadata(document) for document in subset]
    for idx in range(0, len(ids), batch_size):
        batch_end: int = idx + batch_size
        collection.upsert(
            ids=ids[idx:batch_end],
            documents=documents[idx:batch_end],
            metadatas=metadata[idx:batch_end],
        )


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
            add_documents(collection, dataset["train"], *combination)

        collections[combination] = collection

    return collections


if __name__ == "__main__":
    raise NotImplementedError
