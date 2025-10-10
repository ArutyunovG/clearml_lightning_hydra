from omegaconf import DictConfig, OmegaConf
import logging
import os
import pytorch_lightning as pl

from cl_pl_hy.experiment.setup_logging import setup_logging
from cl_pl_hy.experiment.setup_dataloaders import setup_dataloaders
from cl_pl_hy.experiment.utils import import_class
from cl_pl_hy._clearml.task import ClearMLTask
from cl_pl_hy._clearml.dataset import ClearMLDataset
from cl_pl_hy._pytorch_lightning.lit_model import LitModel


def run_experiment(cfg: DictConfig, datasets_dict, clearml_task, pl_loggers=None):
    """Run the actual training experiment."""
    logger = logging.getLogger("cpplhy.experiment")
    
    # Create model
    logger.info("Creating Lightning model...")
    model = LitModel(cfg)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = setup_dataloaders(datasets_dict, cfg.dataloader)
    
    # Get primary dataset name (usually the first one)
    primary_dataset = list(datasets_dict.keys())[0]
    primary_dataloaders = dataloaders.get(primary_dataset, {})
    
    train_loader = primary_dataloaders.get("train")
    val_loader = primary_dataloaders.get("val", primary_dataloaders.get("validation"))
    test_loader = primary_dataloaders.get("test")
    
    # Validate that we have at least a training dataloader
    if train_loader is None:
        raise ValueError(f"No training dataloader found for dataset '{primary_dataset}'. Available splits: {list(primary_dataloaders.keys())}")
    
    logger.info("Dataloaders created:")
    logger.info(f"  Train: {'+' if train_loader else '-'}")
    logger.info(f"  Validation: {'+' if val_loader else '-'}")
    logger.info(f"  Test: {'+' if test_loader else '-'}")
    
    # Get seed from dataloader config
    seed = cfg.dataloader.get("seed", 42)
    pl.seed_everything(seed)

    # Create trainer
    logger.info("Creating PyTorch Lightning Trainer...")
    
    # PyTorch Lightning expects either None, single logger, or list of loggers
    pl_logger = None if not pl_loggers else (pl_loggers[0] if len(pl_loggers) == 1 else pl_loggers)
    
    # Create trainer config (convert to dict to avoid struct mode issues)
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    
    # Handle callbacks if present
    callbacks = []
    if "callbacks" in trainer_cfg:
        callback_configs = trainer_cfg.pop("callbacks")
        for callback_cfg in callback_configs:
            if "class" in callback_cfg:
                callback_cls = import_class(callback_cfg["class"])
                callback_instance = callback_cls(**callback_cfg.get("args", {}))
                callbacks.append(callback_instance)
                logger.info(f"Created callback: {callback_cfg['class']}")
    
    trainer = pl.Trainer(logger=pl_logger, callbacks=callbacks if callbacks else None, **trainer_cfg)
    
    logger.info("Starting training...")
    
    # Train the model
    if val_loader is not None:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
    else:
        logger.info("No validation dataloader found, training without validation")
        trainer.fit(
            model=model,
            train_dataloaders=train_loader
        )
    
    # Test the model if test loader exists
    if test_loader is not None:
        logger.info("Running test...")
        trainer.test(model=model, dataloaders=test_loader)
    
    logger.info("Experiment completed!")


def experiment_main(cfg: DictConfig) -> None:
    """Experiment main, receives config from main.py"""
    logger = logging.getLogger("cpplhy.experiment")
    logger.info("Entered experiment_main")
    pl_loggers = setup_logging(cfg)

    logger.info('Running experiment with config:')
    logger.info('============================================================')
    try:
        # Try to resolve and print the config
        resolved_config = OmegaConf.to_yaml(cfg, resolve=True)
        logger.info(resolved_config)
    except Exception as e:
        # Print unresolved config first, then reraise the exception
        logger.info(OmegaConf.to_yaml(cfg, resolve=False))
        logger.warning(f"Could not fully resolve config for logging: {e}")
        raise
    logger.info('============================================================')
   
    # Initialize ClearML Task with the full config (clearml.yaml is merged into root)
    clearml_task = ClearMLTask(clearml_config=cfg.clearml)

    # Initialize ClearML Dataset manager
    logger.info("Initializing datasets...")
    clearml_dataset = ClearMLDataset(config=cfg.dataset)

    # Create PyTorch datasets from ClearML datasets
    created_datasets = {}
    for dataset_name, info in clearml_dataset.dataset_info.items():
        ds_path = info["path"]
        cls_path = info["class"]
        instances = cfg.dataset[dataset_name].get("instances", [])

        logger.info(
            "Creating dataset '%s': path=%s, class=%s",
            dataset_name, ds_path, cls_path
        )

        DatasetCls = import_class(cls_path)
        created_datasets[dataset_name] = {}

        for inst in instances:
            split_name = inst.get("split", "train")
            args = inst.get("args", {})
            
            # Create dataset instance
            ds_obj = DatasetCls(root_dir=ds_path, **args)
            created_datasets[dataset_name][split_name] = ds_obj
            
            logger.info(
                "Created split '%s' of dataset '%s' with %d samples",
                split_name, dataset_name, len(ds_obj)
            )

    # Run the actual training experiment
    run_experiment(cfg, created_datasets, clearml_task, pl_loggers)
