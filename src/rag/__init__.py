"""RAG module for the relation extraction task.

This module contains functions for the RAG pipeline for the relation
extraction task, namely those that deal with creating, loading, and managing
collections of training documents.

Example:
>>> dataset: DatasetDict = load_dataset("json", data_dir="data")
>>> client: chromadb.Client = chromadb.PersistentClient(path="chroma")
>>> collection: chromadb.Collection = get_collection(dataset, client)
>>> collection: chromadb.Collection = collections[("relation", "subj_type", "obj_type")]
>>> results: dict = collection.query(query_texts=["query text"], n_results=4)
"""

from .collection import get_collection

__all__ = ["get_collection"]
