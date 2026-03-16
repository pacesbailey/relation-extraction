"""
Contains functions for preprocessing the dataset for the relation extraction task.

Functions:
    collate_documents: Collates documents from multiple splits of a dataset into a single list.
    remove_extra_columns: Removes extra columns from a dataset.
    remove_duplicates: Removes duplicates from a dataset.
    preprocess: Preprocesses a dataset by collating documents, splitting into train and test sets, and removing extra columns.
"""

import logging

import pandas as pd
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .conversion import label_document


logger: logging.Logger = logging.getLogger(__name__)


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
    trimmed_dataset: Dataset = dataset.remove_columns(list(extra_columns))
    logger.info(f"Removed {len(dataset) - len(trimmed_dataset)} extra columns")

    return trimmed_dataset


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

    trimmed_dataset: Dataset = Dataset.from_pandas(dataframe)
    logger.info(f"Removed {len(dataset) - len(trimmed_dataset)} duplicate documents")

    return trimmed_dataset


def preprocess(dataset: DatasetDict, format: DictConfig, config: DictConfig) -> DatasetDict:
    """
    Preprocesses a dataset by collating documents, splitting into train and
    test sets, and removing extra columns.

    Args:
        dataset: The dataset to be preprocessed, containing multiple splits.
        format: The format to use for the labeled text.
        config: The configuration containing specifications for the preprocessing.

    Returns:
        The preprocessed dataset, containing the train and test splits.
    """
    # Combine splits, remove superfluous data, prepare labeled input and output
    collated: Dataset = collate_documents(dataset)
    unique: Dataset = remove_duplicates(collated)
    trimmed: Dataset = remove_extra_columns(unique, config.columns)
    labeled: Dataset = trimmed.map(
        function=label_document,
        fn_kwargs={"format": format},
        desc="Labeling documents"
    )
    
    # Split into even train and test splits
    train, test = train_test_split(
        labeled.to_list(),
        random_state=config.random_state,
        stratify=list(labeled["relation"]),
    )
    logger.info(f"Split dataset into {len(train)} train and {len(test)} test documents")
    
    return DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(test)})


if __name__ == "__main__":
    raise NotImplementedError
