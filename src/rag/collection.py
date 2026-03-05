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

from typing import Any

import chromadb
from datasets import Dataset

from .utils import get_metadata


def add_documents(
    collection: chromadb.Collection,
    dataset: Dataset,
    batch_size: int = 5461,
) -> None:
    """
    Adds documents to a collection based on a given relation, subj_type, and
    obj_type.

    Args:
        collection: The collection to add documents to.
        dataset: The dataset to add documents from.
        batch_size: The number of documents to add in each batch.
    """
    ids: list[int] = list(dataset["id"])
    documents: list[str] = list(dataset["text"])
    metadata: list[dict[str, Any]] = [get_metadata(document) for document in dataset]
    for idx in range(0, len(ids), batch_size):
        batch_end: int = idx + batch_size
        collection.upsert(
            ids=ids[idx:batch_end],
            documents=documents[idx:batch_end],
            metadatas=metadata[idx:batch_end],
        )


def get_collection(dataset: Dataset, client: chromadb.Client) -> chromadb.Collection:
    """
    Create collections for the dataset.

    Args:
        dataset: The dataset to create collections for.
        client: The chroma client to use.

    Returns:
        The collection for the dataset.
    """
    collection: chromadb.Collection = client.get_or_create_collection(name="ner")
    if not collection.count():
        add_documents(collection, dataset)

    return collection


if __name__ == "__main__":
    raise NotImplementedError
