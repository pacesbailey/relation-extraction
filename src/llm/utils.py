import re
from enum import StrEnum
from xml.etree.ElementTree import Element, fromstring


TAG_PATTERN: str = r'(</?(?:HEAD|TAIL)(?:\s+ner="([^"]*)")?(?:\s+relation="([^"]*)")?\s*>)'


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
    # Gets the entity token spans
    clean_text: str = re.sub(r"</?(HEAD|TAIL)[^>]*>", "", text).strip()
    tokens: list[str] = clean_text.split()
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


def escape_xml_content(text: str) -> str:
    """Escapes XML special characters in text content, preserving HEAD/TAIL tags.
    
    Args:
        text: The text containing HEAD/TAIL tags and content to escape.

    Returns:
        The text with XML special characters escaped in content portions.
    """
    escaped_parts: list[str] = []
    for part in re.split((pattern := r'(</?(?:HEAD|TAIL)[^>]*>)'), text):
        if re.match(pattern, part):
            escaped_parts.append(part)
        else:
            escaped: str = part.replace('&', '&amp;')
            escaped_parts.append(escaped)
    
    return "".join(escaped_parts)


def extract_xml_data(text: str) -> list[dict]:
    """Extracts the XML data from the text.
    
    Args:
        text: The text to extract the XML data from.

    Returns:
        A list of dictionaries containing the XML data.
    """
    # Extracts data from the XML tags
    escaped_text: str = escape_xml_content(text)
    xml_body: str = f"<root>{re.sub(r"(\w+)='([^']*)'", r'\1="\2"', escaped_text)}</root>"
    root: Element = fromstring(xml_body)
    
    # Organizes the extracted data
    entities: list[dict] = []
    for element in root.iter():
        if element.tag in {Tags.HEAD, Tags.TAIL}:
            try:
                entity_text: str = element.text.strip()
            except AttributeError:
                entity_text = ""
            entity: dict = {
                "tag": element.tag,
                "relation": element.attrib.get("relation"),
                "ner": element.attrib.get("ner"),
                "text": entity_text,
                "start_token": None,
                "end_token": None,
            }
            entities.append(entity)

    return entities


def format_entities(entities: list[dict]) -> dict:
    """Formats the entities into a dictionary.
    
    Args:
        entities: The entities to format.

    Returns:
        A dictionary containing the formatted entities.
    """
    # Reformats the entities into a dictionary
    output: dict = {}
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

def parse_labeled(text: str) -> dict:
    """Parses the labeled text and returns the entities.
    
    Args:
        text: The text to parse.

    Returns:
        A list of dictionaries containing the entities.
    """
    entities: list[dict] = extract_xml_data(text)
    entities = add_token_spans(text, entities)

    return format_entities(entities)


if __name__ == "__main__":
    raise NotImplementedError
