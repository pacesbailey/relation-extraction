from pathlib import Path

import chromadb
import hydra
from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig
from pyrootutils import setup_root

from dataset import preprocess
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
    preprocessed_dataset: DatasetDict = preprocess(dataset, config.dataset.columns, config.dataset.random_state)
    client: chromadb.Client = chromadb.PersistentClient(config.path.chroma)
    collection: chromadb.Collection = get_collection(preprocessed_dataset, client)


if __name__ == "__main__":
    main()
