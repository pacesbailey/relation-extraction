from datasets import Dataset, DatasetDict


def preprocess(dataset: DatasetDict, columns: list[str]) -> DatasetDict:
    formatted_dataset: dict[str, Dataset] = {}
    for split_name, dataset_split in dataset.items():
        all_columns: set[str] = set(dataset_split.column_names)
        extra_columns: set[str] = all_columns - set(columns)
        formatted_split: Dataset = dataset_split.remove_columns(list(extra_columns))
        formatted_dataset[split_name] = formatted_split

    return DatasetDict(formatted_dataset)


if __name__ == "__main__":
    raise NotImplementedError
