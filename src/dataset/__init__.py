"""Dataset module for the relation extraction task."""

from .conversion import label_document, parse_labeled
from .preprocess import preprocess

__all__ = ["label_document", "parse_labeled", "preprocess"]
