from typing import Any


def label_document(document: dict[str, Any]) -> dict[str, Any]:
    """
    Label a document.

    Args:
        document: The document to label.

    Returns:
        The labeled document.
    """
    labeled_tokens: list[str] = []
    for idx, token in enumerate(document["token"]):
        if idx == document["subj_start"]:
            labeled_tokens.append(f"<subj>")
            labeled_tokens.append(token)
        elif idx == document["subj_end"]:
            labeled_tokens.append(token)
            labeled_tokens.append(f"</subj>")
        elif idx == document["obj_start"]:
            labeled_tokens.append(f"<obj>")
            labeled_tokens.append(token)
        elif idx == document["obj_end"]:
            labeled_tokens.append(token)
            labeled_tokens.append(f"</obj>")
        else:
            labeled_tokens.append(token)
    document["labeled_token"] = labeled_tokens

    return document




if __name__ == "__main__":
    raise NotImplementedError
