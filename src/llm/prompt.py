from chromadb import Collection, QueryResult
from omegaconf import DictConfig

from .relation_extractor import RelationExtractor, configure_prompt
from .utils import parse_labeled


def prompt_model(config: DictConfig, document: dict, collection: Collection) -> dict:
    """Prompts the model with the document.
    
    Args:
        config: The configuration for the model.
        document: The document to prompt the model with.
        collection: The collection to prompt the model with.

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
    labeled_text: str = extractor(document["text"])

    return parse_labeled(labeled_text)


if __name__ == "__main__":
    raise NotImplementedError
