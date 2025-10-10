import logging
from typing import Any, Dict, Optional

from omegaconf import DictConfig

import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader


from projects.mnist.src.lit_model import LitModel

logger = logging.getLogger("mnist.train")

def __loader(ds, cfg: DictConfig, train: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=bool(train and cfg.train.shuffle),
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )

def run(cfg: DictConfig, created_datasets: Dict[str, Dict[str, Any]]) -> None:
    pl.seed_everything(cfg.train.seed, workers=True)

    key = "mnist" if "mnist" in created_datasets else next(iter(created_datasets))
    splits = created_datasets[key]

    train_ds = splits["train"]
    val_ds: Optional[torch.utils.data.Dataset] = splits.get("validation")
    test_ds: Optional[torch.utils.data.Dataset] = splits.get("test")

    train_loader = __loader(train_ds, cfg, train=True)
    val_loader = __loader(val_ds, cfg, train=False)

    model = LitModel(cfg.train)

    logger.info("Creating trainer")
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_every_n_steps,
        deterministic=cfg.train.deterministic,
        logger=logging.getLogger("mnist.train.pl")
    )
    logger.info("Start training")
    trainer.fit(model, train_loader, val_loader)
    logger.info("Training complete")

