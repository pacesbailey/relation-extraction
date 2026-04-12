"""
Contains functions for prompting the model for the relation extraction task.

Functions:
    prompt_model: Prompts the model for the relation extraction task.
"""

from copy import deepcopy

from chromadb import Collection, QueryResult
from omegaconf import DictConfig

from .relation_extractor import RelationExtractor, configure_prompt
from .utils import parse_labeled


def prompt_model(document: dict, collection: Collection, config: DictConfig) -> dict:
    """Prompts the model with the document.
    
    Args:
        document: The document to prompt the model with.
        collection: The collection to prompt the model with.
        config: The configuration for the model.

    Returns:
        The entities predicted by the model.
    """
    document_copy: dict = deepcopy(document)
    filter: dict = {
        "$and": [
            {"relation": document_copy["relation"]},
            {"subj_type": document_copy["subj_type"]},
            {"obj_type": document_copy["obj_type"]}
        ]
    }
    examples: QueryResult = collection.query(
        query_texts=document_copy["text"],
        n_results=config.rag.n_queries,
        where=filter
    )
    prompt: str = configure_prompt(config, examples, document_copy)
    extractor: RelationExtractor = RelationExtractor(prompt)
    response: str = extractor(document_copy["text"])
    document_copy["response"] = response

    return parse_labeled(document_copy)


if __name__ == "__main__":
    raise NotImplementedError
