import importlib
from omegaconf import DictConfig, OmegaConf
import logging


from cl_pl_hy.experiment.setup_logging  import setup_logging
from cl_pl_hy._clearml.task import ClearMLTask
from cl_pl_hy._clearml.dataset import ClearMLDataset


def __load_class(dotted_path: str):
    """
    dotted_path: e.g. "projects.mnist.MNISTDataset"
    """
    module_name, cls_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


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

    created_datasets = {}
    for dataset_name, info in clearml_dataset.dataset_info.items():
        ds_path = info.get("path") or info.get("local_copy_dir")
        cls_path = info["class"]
        instances = cfg.dataset[dataset_name].get("instances", [])

        logger.info(
            "Trying to create dataset '%s': path=%s, class=%s",
            dataset_name, ds_path, cls_path
        )

        DatasetCls = __load_class(cls_path)
        created_datasets[dataset_name] = {}

        logger.info("Dataset class loaded: %s", DatasetCls)
        logger.info("Creating instances {}".format(instances))

        for inst in instances:
            split_name = inst.get("split") or "train"
            args = inst.get("args", {})  # kwargs for your dataset class
            # Common pattern: pass the local root plus any extra args
            ds_obj = DatasetCls(root_dir=ds_path, **args)
            created_datasets[dataset_name][split_name] = ds_obj
            logger.info(
                "  Created split '%s' of dataset '%s' with %d samples",
                split_name, dataset_name, len(ds_obj)
            )
