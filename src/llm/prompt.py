"""
Contains functions for prompting the model for the relation extraction task.

Functions:
    prompt_model: Prompts the model for the relation extraction task.
"""

from chromadb import Collection, QueryResult
from omegaconf import DictConfig

from .relation_extractor import RelationExtractor, configure_prompt


def prompt_model(document: dict, collection: Collection, config: DictConfig) -> dict:
    """Prompts the model with the document.
    
    Args:
        document: The document to prompt the model with.
        collection: The collection to prompt the model with.
        config: The configuration for the model.

    Returns:
        The entities predicted by the model.
    """
    filter: dict = {
        "$and": [
            {"relation": document["relation"]},
            {"subj_type": document["subj_type"]},
            {"obj_type": document["obj_type"]}
        ]
    }
    examples: QueryResult = collection.query(
        query_texts=document["text"],
        n_results=config.rag.n_queries,
        where=filter
    )
    prompt: str = configure_prompt(config, examples, document)
    extractor: RelationExtractor = RelationExtractor(prompt)

    return {"response": extractor(document["text"])}


if __name__ == "__main__":
    raise NotImplementedError
