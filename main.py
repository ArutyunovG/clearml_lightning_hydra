import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


PROJECT = os.getenv("CPPLHY_PROJECT", "mnist")
PROJECT = os.path.join(REPO_ROOT, "projects", PROJECT)

MAIN_CONFIG_PATH = os.getenv("CPPLHY_CONFIG_PATH", "main")


@hydra.main(config_path=PROJECT, 
            config_name="main",
            version_base=None)
def main(cfg: DictConfig) -> None:

    # Call experiment_main with the config
    from cl_pl_hy.experiment.experiment_main import experiment_main
    experiment_main(cfg)


if __name__ == "__main__":

    # Initial basic logging setup (will be reconfigured in main() based on config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("cpplhy.main")

    import cl_pl_hy  as _
    logger.info("cl_pl_hy imported successfully")

    main()
