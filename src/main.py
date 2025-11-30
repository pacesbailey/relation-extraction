from pathlib import Path

import hydra
from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig
from pyrootutils import setup_root

from preprocess import preprocess


root: Path = setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True)

@hydra.main(config_path=str(root / "config"), config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    """Main function for the relation extraction task.

    Args:
        config: The configuration containing specifications for the relation extraction task.
    """
    dataset: DatasetDict = load_dataset(config.dataset.path, data_dir=config.dataset.data_dir)
    preprocessed_dataset: DatasetDict = preprocess(dataset, config)


if __name__ == "__main__":
    main()
