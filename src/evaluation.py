import json
from itertools import chain
from pathlib import Path

from datasets import Dataset
from sklearn.metrics import classification_report


def group_classification_reports(dataset: Dataset) -> dict:
    """Groups the classification reports.
    
    Args:
        reports: The classification reports to group.

    Returns:
        A dictionary containing the grouped classification reports.
    """
    return {
        "subj_type": classification_report(dataset["subj_type"], dataset["pred_subj_type"], output_dict=True, zero_division=0),
        "subj_spans": evaluate_spans(dataset, "SUBJ"),
        "obj_type": classification_report(dataset["obj_type"], dataset["pred_obj_type"], output_dict=True, zero_division=0),
        "obj_spans": evaluate_spans(dataset, "OBJ"),
        "relation": classification_report(dataset["relation"], dataset["pred_relation"], output_dict=True, zero_division=0)
    }


def evaluate(dataset: Dataset, path: Path) -> None:
    """Evaluates the dataset.
    
    Args:
        dataset: The dataset to evaluate.
    """
    classification_reports: dict = group_classification_reports(dataset)
    report_evaluation(classification_reports)
    save_evaluation(classification_reports, path)


def evaluate_spans(dataset: Dataset, prefix: str) -> dict:
    """Evaluates the spans of the entities.
    
    Args:
        dataset: The dataset to evaluate the spans of the entities.
        prefix: The prefix of the entities.

    Returns:
        A dictionary containing the evaluation of the spans of the entities.
    """
    true_bios: list[list[str]] = []
    pred_bios: list[list[str]] = []
    for document in dataset:
        n_tokens: int = len(document["token"])
        true_bio: list[str] = span_to_bio(
            n_tokens=n_tokens,
            start=document[f"{prefix.lower()}_start"],
            end=document[f"{prefix.lower()}_end"],
            prefix=prefix
        )
        pred_bio: list[str] = span_to_bio(
            n_tokens=n_tokens,
            start=document[f"pred_{prefix.lower()}_start"],
            end=document[f"pred_{prefix.lower()}_end"],
            prefix=prefix
        )
        true_bios.append(true_bio)
        pred_bios.append(pred_bio)

    return classification_report(
        y_true=list(chain.from_iterable(true_bios)),
        y_pred=list(chain.from_iterable(pred_bios)),
        labels=[f"B-{prefix}", f"I-{prefix}"],
        output_dict=True,
        zero_division=0
    )


def report_evaluation(classification_reports: dict) -> None:
    """Reports the evaluation.
    
    Args:
        classification_reports: The classification reports to report.
    """
    print(classification_reports)


def save_evaluation(classification_reports: dict, path: Path) -> None:
    """Saves the evaluation.
    
    Args:
        classification_reports: The classification reports to save.
        path: The path to save the evaluation.
    """
    with open(path, "w", encoding="utf-8") as file:
        for key, value in classification_reports.items():
            json.dump({key: value}, file)
            file.write("\n")


def span_to_bio(n_tokens: int, start: int, end: int, prefix: str) -> list[str]:
    """Converts a span to a BIO tag sequence.
    
    Args:
        n_tokens: The number of tokens in the sequence.
        start: The start index of the span.
        end: The end index of the span.
        prefix: The prefix to use for the BIO tags.

    Returns:
        A list of BIO tags.
    """
    tags: list[str] = ["O"] * n_tokens
    if start is None or end is None:
        return tags

    for i in range(start, end + 1):
        tags[i] = f"I-{prefix}"
    tags[start] = f"B-{prefix}"

    return tags


if __name__ == "__main__":
    raise NotImplementedError
