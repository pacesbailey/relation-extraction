import re
from enum import StrEnum


TAG_PATTERN: re.Pattern = re.compile(r"<(HEAD|TAIL)\s+ner=['\"]([^'\"]+)['\"]\s+relation=['\"]([^'\"]+)['\"]\s*>([\s\S]*?)</\1>", re.IGNORECASE)
CLEAN_PATTERN: re.Pattern = re.compile(r"</?(HEAD|TAIL)[^>]*>", re.IGNORECASE)


class Tags(StrEnum):
    HEAD = "HEAD"
    TAIL = "TAIL"


def check_overlap(first_span: tuple[int, int], second_span: tuple[int, int]) -> bool:
    """Checks if two spans overlap.
    
    Args:
        first_span: The first span to check.
        second_span: The second span to check.

    Returns:
        True if the spans overlap, False otherwise.
    """
    first_start, first_end = first_span
    second_start, second_end = second_span

    return not (first_end < second_start or second_end < first_start)


def add_token_spans(entities: list[dict], reference_tokens: list[str]) -> list[dict]:
    """Adds the token spans to the entities.
    
    Args:
        entities: The entities to add the token spans to.
        reference_tokens: The reference tokens to add the token spans to.

    Returns:
        The entities with the token spans added.
    """
    used_spans: list[tuple[int, int]] = []
    for entity in entities:
        entity_tokens: list[str] = entity["text"].split()
        token_length: int = len(entity_tokens)
        if token_length == 0 or token_length > len(reference_tokens):
            continue
        for idx in range(0, len(reference_tokens) - token_length + 1):
            if reference_tokens[idx : idx + token_length] != entity_tokens:
                continue
            span: tuple[int, int] = (idx, idx + token_length - 1)
            if any(check_overlap(span, used) for used in used_spans):
                continue
            entity["start_token"] = idx
            entity["end_token"] = span[1]
            used_spans.append(span)
            break

    return entities


def format_entities(entities: list[dict]) -> dict:
    """Formats the entities into a dictionary.
    
    Args:
        entities: The entities to format.

    Returns:
        A dictionary containing the formatted entities.
    """
    output: dict = {}
    for tag in "obj", "subj":
        output[f"pred_{tag}_type"] = "UNPARSED"
        output[f"pred_{tag}_start"] = None
        output[f"pred_{tag}_end"] = None

    relations: set[str] = set()
    for entity in entities:
        relations.add(entity["relation"])
        match entity["tag"]:
            case Tags.HEAD:
                output["pred_subj_type"] = entity["ner"]
                output["pred_subj_start"] = entity["start_token"]
                output["pred_subj_end"] = entity["end_token"]
            case Tags.TAIL:
                output["pred_obj_type"] = entity["ner"]
                output["pred_obj_start"] = entity["start_token"]
                output["pred_obj_end"] = entity["end_token"]
            case _:
                raise ValueError(f"Invalid entity tag: {entity['tag']}")

    output["pred_relation"] = relations.pop() if len(relations) == 1 else "no_relation"
    
    return output


def parse_entities(text: str) -> list[dict]:
    """Parses the entities from the text, yielding the head/tail tags, the NER
    type, the relation type, and the entity text.
    
    Args:
        text: The text to parse the entities from.
    
    Returns:
        A list of dictionaries containing the entities.
    """
    entities: list[dict] = []
    for labeled_entity in TAG_PATTERN.finditer(text):
        tag, ner, relation, entity = labeled_entity.groups()
        entity_tags: dict = {
            "tag": tag,
            "ner": ner,
            "relation": relation,
            "text": entity.strip(),
            "start_token": None,
            "end_token": None
        }
        entities.append(entity_tags)
    
    return entities


def parse_labeled(document: dict) -> dict:
    """Parses the labeled text and returns the entities.
    
    Args:
        document: Must include ``response`` (model output) and ``token`` (dataset tokens).

    Returns:
        Prediction fields aligned to ``document["token"]`` indices.
    """
    text: str = re.sub(r"\\/", "/", document["response"])
    entities: list[dict] = parse_entities(text)
    entities = add_token_spans(entities, document["token"])

    return format_entities(entities)


if __name__ == "__main__":
    raise NotImplementedError
