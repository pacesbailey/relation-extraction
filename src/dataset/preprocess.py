"""Preprocessing module for the tacred dataset.

This module contains functions for preprocessing the tacred dataset in
preparation for the relation extraction task. The preprocessing pipeline
includes the following steps:
1. Collating documents from multiple splits of the dataset into a single list.
2. Removing extra columns from the collated documents.
3. Splitting the collated documents into train and test sets.
4. Formatting the train and test sets into Dataset objects.
5. Returning the preprocessed dataset as a DatasetDict.

Example:
>>> dataset: DatasetDict = load_dataset("json", data_dir="data")
>>> preprocessed_dataset: DatasetDict = preprocess(dataset, config)
"""

import pandas as pd
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def collate_documents(dataset: DatasetDict) -> list[dict]:
    """
    Collates documents from multiple splits of a dataset into a single list.

    Args:
        dataset: A DatasetDict containing the dataset splits to collate.

    Returns:
        A list of documents.
    """
    collated_documents: list[dict] = []
    for dataset_split in dataset.values():
        for document in dataset_split:
            collated_documents.append(document)

    return collated_documents


def remove_extra_columns(dataset: Dataset, columns: list[str]) -> Dataset:
    """
    Removes extra columns from a dataset.

    Args:
        dataset: The dataset split to remove columns from.
        columns: A list of columns to keep.

    Returns:
        The dataset split with the extra columns removed.
    """
    all_columns: set[str] = set(dataset.column_names)
    extra_columns: set[str] = all_columns - set(columns)

    return dataset.remove_columns(list(extra_columns))


def remove_duplicates(dataset: Dataset) -> Dataset:
    """
    Removes duplicates from a dataset.

    Args:
        dataset: The dataset to remove duplicates from.

    Returns:
        The dataset with duplicates removed.
    """
    dataframe: pd.DataFrame = pd.DataFrame(dataset)
    dataframe = dataframe.drop_duplicates(subset=["token"])

    return Dataset.from_pandas(dataframe)


def preprocess(dataset: DatasetDict, config: DictConfig) -> DatasetDict:
    """
    Preprocesses a dataset by collating documents, splitting into train and
    test sets, and removing extra columns.

    Args:
        dataset: The dataset to be preprocessed, containing multiple splits.
        config: The configuration containing specifications for the
            preprocessing pipeline.

    Returns:
        The preprocessed dataset, containing the train and test splits.
    """
    collated_documents: list[dict] = collate_documents(dataset)
    collated_dataset: Dataset = Dataset.from_list(collated_documents)
    collated_dataset = remove_duplicates(collated_dataset)
    collated_documents = list(collated_dataset)
    train_documents, test_documents = train_test_split(
        collated_documents,
        random_state=config.dataset.random_state,
        stratify=[doc["relation"] for doc in collated_documents],
    )
    formatted_dataset: dict[str, Dataset] = {}
    for split_name, documents in zip(
        ["train", "test"],
        [train_documents, test_documents]
    ):
        split_subset: Dataset = Dataset.from_list(documents)
        split_subset = remove_extra_columns(split_subset, config.dataset.columns)
        split_subset = split_subset.map(lambda x: {"text": " ".join(x["token"])})
        formatted_dataset[split_name] = split_subset

    return DatasetDict(formatted_dataset)


if __name__ == "__main__":
    raise NotImplementedError
