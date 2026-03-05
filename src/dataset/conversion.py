from enum import StrEnum
from typing import Any


class Tags(StrEnum):
    HEAD: str = "HEAD"
    TAIL: str = "TAIL"


def group_spans(document: dict[str, Any]) -> list[dict[str, Any]]:
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

def insert_entity_tags(tokens: list[str], spans: list[dict[str, Any]]) -> list[str]:
    labeled_tokens: list[str] = tokens.copy()
    for i, span in enumerate(spans):
        labeled_tokens.insert(span["end"] + 1, f"</{span['tag']}>")
        labeled_tokens.insert(span["start"], f"<{span['tag']} relation='{span['relation']}' ner='{span['ner']}'>")
        span["start"] += 1
        span["end"] += 1
        for j in range(i):
            spans[j]["start"] += 2
            spans[j]["end"] += 2

    return labeled_tokens


def label_document(document: dict[str, Any]) -> dict[str, str]:
    spans: list[dict[str, Any]] = group_spans(document)
    labeled_tokens: list[str] = insert_entity_tags(document["token"], spans)
    return {"labeled_text": " ".join(labeled_tokens)}


if __name__ == "__main__":
    raise NotImplementedError
