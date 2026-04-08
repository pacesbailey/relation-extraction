import logging
from pathlib import Path

import dspy
import hydra
from chromadb import ClientAPI, Collection, PersistentClient
from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig
from pyrootutils import setup_root

from dataset import preprocess
from evaluation import evaluate
from llm import prompt_model
from rag import get_collection


root: Path = setup_root(__file__)
logger: logging.Logger = logging.getLogger(__name__)


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
    test_subset: Dataset = dataset["test"].shuffle(config.dataset.random_state).select(range(20))
    client: ClientAPI = PersistentClient(config.path.chroma)
    collection: Collection = get_collection(dataset["train"], client)
    
    # Loads, configures, and prompts the model
    lm: dspy.LM = dspy.LM(**config.model)
    dspy.configure(lm=lm)
    predictions: Dataset = test_subset.map(prompt_model, fn_kwargs={"collection": collection, "config": config})
    predictions.to_json(config.path.predictions)

    evaluate(predictions, config.path.scores)


if __name__ == "__main__":
    main()
