import pandas as pd
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
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
    format: DictConfig,
    dataset_config: DictConfig,
) -> DatasetDict:
    """
    Preprocesses a dataset by collating documents, splitting into train and
    test sets, and removing extra columns.

    Args:
        dataset: The dataset to be preprocessed, containing multiple splits.
        column_names: The columns to keep in the dataset.
        random_state: The random state to use for the train-test split.
        format: The format to use for the labeled text.

    Returns:
        The preprocessed dataset, containing the train and test splits.
    """
    collated_dataset: Dataset = collate_documents(dataset)
    unique_dataset: Dataset = remove_duplicates(collated_dataset)
    trimmed_dataset: Dataset = remove_extra_columns(unique_dataset, dataset_config.columns)
    text_dataset: Dataset = trimmed_dataset.map(
        function=lambda x: {"text": " ".join(x["token"])},
        desc="Joining tokens"
    )
    train_documents, test_documents = train_test_split(
        text_dataset.to_list(),
        random_state=dataset_config.random_state,
        stratify=list(text_dataset["relation"]),
    )
    train_dataset: Dataset = Dataset.from_list(train_documents)
    test_dataset: Dataset = Dataset.from_list(test_documents)
    datasets: dict[str, Dataset] = {
        "train": train_dataset.map(label_document, fn_kwargs={"format": format}, desc="Labeling documents"),
        "test": test_dataset.map(label_document, fn_kwargs={"format": format}, desc="Labeling documents")
    }

    return DatasetDict(datasets)


if __name__ == "__main__":
    raise NotImplementedError
