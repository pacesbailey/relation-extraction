from pathlib import Path

import dspy
import hydra
from chromadb import Client, Collection, PersistentClient
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig
from pyrootutils import setup_root

from dataset import preprocess
from llm import prompt_model
from rag import get_collection


root: Path = setup_root(__file__)


@hydra.main(config_path=str(root / "config"), config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main function for the relation extraction task.

    Args:
        config: The configuration containing specifications for the relation
            extraction task.
    """
    # Loads, preprocesses, and creates collections for the dataset
    dataset: DatasetDict = load_dataset(config.dataset.path, data_dir=config.dataset.data_dir)
    dataset = preprocess(dataset, config.prompt.format, config.dataset)
    client: Client = PersistentClient(config.path.chroma)
    collection: Collection = get_collection(dataset["train"], client)
    
    # Loads, configures, and prompts the model
    lm: dspy.LM = dspy.LM(**config.model)
    dspy.configure(lm=lm)
    predictions: Dataset = dataset["test"].map(lambda document: prompt_model(config, document, collection))
    predictions.to_json(config.path.predictions / "predictions.json")


if __name__ == "__main__":
    main()
