import re


def clean_string(text: str, replacement: str = "_") -> str:
    """
    Clean a string by replacing all non-alphanumeric characters with a
    replacement character.

    Args:
        text: The string to clean.
        replacement: The replacement character.

    Returns:
        The cleaned string.
    """
    return re.sub(r"[^a-zA-Z0-9]", replacement, text)


def get_metadata(document: dict) -> dict:
    """
    Get the metadata for a document.

    Args:
        document: The document to get metadata for.

    Returns:
        The metadata for the document.
    """
    return {
        "relation": document["relation"],
        "subj_type": document["subj_type"],
        "obj_type": document["obj_type"],
        "input": document["input"],
        "output": document["output"],
    }


if __name__ == "__main__":
    raise NotImplementedError
