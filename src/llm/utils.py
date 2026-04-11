import re
from enum import StrEnum


TAG_PATTERN: re.Pattern = re.compile(r"<(HEAD|TAIL)\s+ner=['\"]([^'\"]+)['\"]\s+relation=['\"]([^'\"]+)['\"]\s*>([\s\S]*?)</\1>", re.IGNORECASE)
CLEAN_PATTERN: re.Pattern = re.compile(r"</?(HEAD|TAIL)[^>]*>", re.IGNORECASE)


class Tags(StrEnum):
    HEAD = "HEAD"
    TAIL = "TAIL"


def add_token_spans(text: str, entities: list[dict]) -> list[dict]:
    """Adds the token spans to the entities.
    
    Args:
        text: The text to add the token spans to.
        entities: The entities to add the token spans to.

    Returns:
        A list of dictionaries containing the entities with the token spans added.
    """
    tokens: list[str] = CLEAN_PATTERN.sub("", text).strip().split()
    search_start: int = 0
    for entity in entities:
        entity_tokens: list[str] = entity["text"].split()
        token_length: int = len(entity_tokens)
        for idx in range(search_start, len(tokens) - token_length + 1):
            if tokens[idx : idx + token_length] == entity_tokens:
                entity["start_token"] = idx
                entity["end_token"] = idx + token_length - 1
                search_start = entity["end_token"] + 1
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

    match len(relations):
        case 0:
            output["pred_relation"] = "no_relation"
        case 1:
            output["pred_relation"] = relations.pop()
        case _:
            raise ValueError(f"Expected 1 relation, got multiple: {relations}")
    
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
        document: The document to parse.

    Returns:
        A list of dictionaries containing the entities.
    """
    text: str = re.sub(r"\\/", "/", document["response"])
    entities: list[dict] = parse_entities(text)
    entities = add_token_spans(text, entities)

    return format_entities(entities)


if __name__ == "__main__":
    raise NotImplementedError
