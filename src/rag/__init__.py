"""
Contains functions for creating and managing collections of documents for the
relation extraction task.

Functions:
    get_collection: Retrieves or creates a collection for the dataset.
"""

from .collection import get_collection

__all__ = ["get_collection"]
