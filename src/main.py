from pathlib import Path

import chromadb
import hydra
from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig
from pyrootutils import setup_root

from dataset import preprocess
from rag import Combination, get_collections


root: Path = setup_root(
    __file__, project_root_env_var=True, dotenv=True, pythonpath=True
)


@hydra.main(
    config_path=str(root / "config"), config_name="config.yaml", version_base=None
)
def main(config: DictConfig) -> None:
    """Main function for the relation extraction task.

    Args:
        config: The configuration containing specifications for the relation
            extraction task.
    """
    dataset: DatasetDict = load_dataset(
        config.dataset.path, data_dir=config.dataset.data_dir
    )
    preprocessed_dataset: DatasetDict = preprocess(dataset, config)
    client: chromadb.Client = chromadb.PersistentClient(config.path.chroma)
    collections: dict[Combination, chromadb.Collection] = get_collections(
        preprocessed_dataset, client
    )
    test_doc: dict = preprocessed_dataset["test"][0]
    relation: str = test_doc["relation"]
    subj_type: str = test_doc["subj_type"]
    obj_type: str = test_doc["obj_type"]
    collection: chromadb.Collection = collections[(relation, subj_type, obj_type)]
    results: dict = collection.query(query_texts=[test_doc["text"]], n_results=4)
    print(results)


if __name__ == "__main__":
    main()
