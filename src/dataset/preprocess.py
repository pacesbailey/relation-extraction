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
from sklearn.model_selection import train_test_split

from .conversion import label_document


def collate_documents(dataset: DatasetDict) -> Dataset:
    """
    Collates documents from multiple splits of a dataset into a single list.

    Args:
        dataset: A DatasetDict containing the dataset splits to collate.

    Returns:
        A Dataset containing the collated documents.
    """
    documents: list[dict] = [
        document for split in dataset.values()
        for document in split.to_list()
    ]
    
    return Dataset.from_list(documents)


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


def preprocess(
    dataset: DatasetDict,
    column_names: list[str],
    random_state: int,
) -> DatasetDict:
    """
    Preprocesses a dataset by collating documents, splitting into train and
    test sets, and removing extra columns.

    Args:
        dataset: The dataset to be preprocessed, containing multiple splits.
        column_names: The columns to keep in the dataset.
        random_state: The random state to use for the train-test split.

    Returns:
        The preprocessed dataset, containing the train and test splits.
    """
    collated_dataset: Dataset = collate_documents(dataset)
    unique_dataset: Dataset = remove_duplicates(collated_dataset)
    trimmed_dataset: Dataset = remove_extra_columns(unique_dataset, column_names)
    text_dataset: Dataset = trimmed_dataset.map(lambda x: {"text": " ".join(x["token"])})
    formatted_documents: list[dict] = text_dataset.to_list()
    train_documents, test_documents = train_test_split(
        formatted_documents,
        random_state=random_state,
        stratify=[doc["relation"] for doc in formatted_documents],
    )
    formatted_dataset: dict[str, Dataset] = {
        "train": Dataset.from_list(train_documents),
        "test": Dataset.from_list(test_documents)
    }
    formatted_dataset["train"] = formatted_dataset["train"].map(label_document)

    return DatasetDict(formatted_dataset)


if __name__ == "__main__":
    raise NotImplementedError
