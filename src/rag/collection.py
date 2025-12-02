import re

import chromadb
from datasets import Dataset, DatasetDict


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


def get_collections(dataset: DatasetDict, client: chromadb.Client) -> dict[Combination, chromadb.Collection]:
    """
    Create collections for the dataset.

    Args:
        dataset: The dataset to create collections for.
        client: The chroma client to use.

    Returns:
        A dictionary containing the collections by combination.
    """
    # Filter the dataset for unique relation, subj_type, and obj_type combinations
    collections: dict[Combination, chromadb.Collection] = {}
    for relation in set(dataset["train"]["relation"]):
        relation_subset: Dataset = dataset["train"].filter(lambda x: x["relation"] == relation)
        for subj_type in set(relation_subset["subj_type"]):
            subj_subset: Dataset = relation_subset.filter(lambda x: x["subj_type"] == subj_type)
            for obj_type in set(subj_subset["obj_type"]):
                obj_subset: Dataset = subj_subset.filter(lambda x: x["obj_type"] == obj_type)
                combination: Combination = (relation, subj_type, obj_type)
                cleaned_name: str = clean_string("-".join(combination))
                collection: chromadb.Collection = client.get_or_create_collection(name=cleaned_name)
                if collection.count() == 0:  # If the collection is empty, add the documents
                    ids: list[int] = list(obj_subset["id"])
                    docs: list[str] = list(obj_subset["text"])
                    for idx in range(0, len(ids), (max := 5461)):
                        collection.add(ids=ids[idx:idx+max], documents=docs[idx:idx+max])
                collections[combination] = collection

    return collections


if __name__ == "__main__":
    raise NotImplementedError
