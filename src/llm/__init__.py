"""
Contains modules for the relation extraction task.

Modules:
    prompt: Contains functions for prompting the model for the relation extraction task.
"""

from .prompt import prompt_model
from .utils import parse_labeled

__all__ = ["parse_labeled", "prompt_model"]
