"""
Contains functions for preprocessing the dataset for the relation extraction task.

Modules:
    label_document: Labels the document with the entity tags.
    preprocess: Preprocesses a dataset by collating documents, splitting into train and test sets, and removing extra columns.
"""

from .conversion import label_document
from .preprocess import preprocess

__all__ = ["label_document", "preprocess"]
