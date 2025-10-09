from omegaconf import DictConfig, OmegaConf
import logging

from cl_pl_hy.experiment.setup_logging  import setup_logging
from cl_pl_hy._clearml.task import ClearMLTask
from cl_pl_hy._clearml.dataset import ClearMLDataset

def experiment_main(cfg: DictConfig) -> None:
    """Experiment main, receives config from main.py"""
    logger = logging.getLogger("cpplhy.experiment")
    logger.info("Entered experiment_main")
    setup_logging(cfg)

    logger.info('Running experiment with config:')
    logger.info('============================================================')
    logger.info(f'{OmegaConf.to_yaml(cfg, resolve=True)}')
    logger.info('============================================================')
   
    # Initialize ClearML Task with the full config (clearml.yaml is merged into root)
    clearml_task = ClearMLTask(clearml_config=cfg.clearml)

    # Initialize ClearML Dataset manager
    logger.info("Initializing datasets...")
    clearml_dataset = ClearMLDataset(config=cfg.dataset)

    for dataset_name, info in clearml_dataset.dataset_info.items():
        logger.info("Trying to create dataset '%s': path=%s, class=%s", dataset_name, info['path'], info['class'])
