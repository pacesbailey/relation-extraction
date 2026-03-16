import re
from copy import deepcopy
from enum import StrEnum
from xml.etree.ElementTree import Element, fromstring

from omegaconf import DictConfig


class Tags(StrEnum):
    HEAD: str = "HEAD"
    TAIL: str = "TAIL"


def group_spans(document: dict) -> list[dict]:
    """Groups the spans of the head and tail entities in the document.
    
    Args:
        document: The document to group the spans of the head and tail entities.

    Returns:
        A list of dictionaries containing the spans of the head and tail entities.
    """
    head_span: dict = {
        "tag": Tags.HEAD,
        "relation": document["relation"],
        "ner": document["subj_type"],
        "start": document["subj_start"],
        "end": document["subj_end"]
    }
    tail_span: dict = {
        "tag": Tags.TAIL,
        "relation": document["relation"],
        "ner": document["obj_type"],
        "start": document["obj_start"],
        "end": document["obj_end"]
    }

    return sorted([tail_span, head_span], key=lambda s: -s["end"])


def insert_entity_tags(tokens: list[str], spans: list[dict], format: str) -> list[str]:
    """Inserts the entity tags into the tokens.
    
    Args:
        tokens: The tokens to insert the entity tags into.
        spans: The spans of the head and tail entities.
        format: The format to use for the labeled text.

    Returns:
        A list of strings with the entity tags inserted.
    """
    beginning, ending = tuple(format.split("{entity}"))
    if not beginning and not ending:
        return tokens

    spans_copy: list[dict] = deepcopy(spans)
    labeled_tokens: list[str] = tokens.copy()
    for i, span in enumerate(spans_copy):
        labeled_tokens.insert(span["end"] + 1, ending.format(**span))
        labeled_tokens.insert(span["start"], beginning.format(**span))
        span["start"] += 1
        span["end"] += 1
        for j in range(i):
            spans_copy[j]["start"] += 2
            spans_copy[j]["end"] += 2

    return labeled_tokens


def label_document(document: dict, format: DictConfig) -> dict[str, str]:
    """Labels the document with the entity tags.
    
    Args:
        document: The document to label with the entity tags.
        format: The format to use for the labeled text.

    Returns:
        A dictionary containing the labeled text.
    """
    spans: list[dict] = group_spans(document)
    input_tokens: list[str] = insert_entity_tags(document["token"], spans, format.input)
    output_tokens: list[str] = insert_entity_tags(document["token"], spans, format.output)
    
    return {
        "text": " ".join(document["token"]),
        "input": " ".join(input_tokens),
        "output": " ".join(output_tokens)
    }


if __name__ == "__main__":
    raise NotImplementedError
