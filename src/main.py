from pathlib import Path

import hydra
from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig
from pyrootutils import setup_root

from preprocess import preprocess


root: Path = setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True)

@hydra.main(config_path=str(root / "config"), config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    dataset: DatasetDict = load_dataset(cfg.dataset.path, data_dir=cfg.dataset.data_dir)
    dataset = preprocess(dataset, cfg.dataset.columns)


if __name__ == "__main__":
    main()
