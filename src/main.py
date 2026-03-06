from pathlib import Path

import hydra
from chromadb import Client, Collection, PersistentClient, QueryResult
from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig
from pyrootutils import setup_root

import dspy
from dataset import preprocess
from llm import RelationExtractor, configure_prompt
from rag import get_collection


root: Path = setup_root(search_from=str(__file__), pythonpath=True)


@hydra.main(config_path=str(root / "config"), config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main function for the relation extraction task.

    Args:
        config: The configuration containing specifications for the relation
            extraction task.
    """
    dataset: DatasetDict = load_dataset(config.dataset.path, data_dir=config.dataset.data_dir)
    dataset = preprocess(dataset, config.dataset.columns, config.dataset.random_state)
    client: Client = PersistentClient(config.path.chroma)
    collection: Collection = get_collection(dataset["train"], client)
    lm: dspy.LM = dspy.LM(**config.model)
    dspy.configure(lm=lm)
    for document in dataset["test"]:
        filter: dict = {
            "$and": [
                {"relation": document["relation"]},
                {"subj_type": document["subj_type"]},
                {"obj_type": document["obj_type"]}
            ]
        }
        examples: QueryResult = collection.query(query_texts=document["text"], n_results=2, where=filter)
        prompt: str = configure_prompt(config, examples, document)
        extractor: RelationExtractor = RelationExtractor(prompt)
        labeled_text: str = extractor(document["text"])
        print(f"{prompt=}")
        print(f"{document['text']=}")
        print(f"{labeled_text=}")
        exit()


if __name__ == "__main__":
    main()
