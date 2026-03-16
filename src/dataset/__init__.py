"""Dataset module for the relation extraction task."""

from .conversion import label_document
from .preprocess import preprocess

__all__ = ["label_document", "preprocess"]
