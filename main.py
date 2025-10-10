import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf


# Make the repo package importable (so pkg://cl_pl_hy.conf works)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PROJECT = os.getenv("CPPLHY_PROJECT", "mnist")
CONFIG_PATH = os.path.join(REPO_ROOT, "projects", PROJECT)
CONFIG_NAME = os.getenv("CPPLHY_CONFIG_NAME", "main")

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    from cl_pl_hy.experiment.experiment_main import experiment_main
    experiment_main(cfg)


if __name__ == "__main__":

    # Initial basic logging setup - console only
    # (full logging will be reconfigured in experiment_main() based on config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()]  # Only console handler, no file
    )
    logger = logging.getLogger("cpplhy.main")

    import cl_pl_hy  as _
    logger.info("cl_pl_hy imported successfully")

    main()
