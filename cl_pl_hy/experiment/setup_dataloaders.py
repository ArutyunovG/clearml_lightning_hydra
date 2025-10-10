"""
DataLoader setup functionality for experiments.
"""
import logging
from torch.utils.data import DataLoader


def setup_dataloaders(datasets_dict, dataloader_cfg):
    """Create PyTorch DataLoaders from datasets.
    
    Args:
        datasets_dict: Dictionary of datasets organized by dataset_name -> split_name -> dataset
        dataloader_cfg: Configuration for DataLoader parameters
        
    Returns:
        Dictionary of dataloaders organized by dataset_name -> split_name -> dataloader
    """
    logger = logging.getLogger("cpplhy.experiment.setup_dataloaders")
    dataloaders = {}
    
    for dataset_name, splits in datasets_dict.items():
        dataloaders[dataset_name] = {}
        
        for split_name, dataset in splits.items():
            # Get appropriate dataloader config for this split
            if split_name == "train":
                dl_args = dataloader_cfg.get("args", {})
            elif split_name in ["val", "validation"]:
                dl_args = dataloader_cfg.get("val_args", dataloader_cfg.get("args", {}))
            elif split_name == "test":
                dl_args = dataloader_cfg.get("test_args", dataloader_cfg.get("args", {}))
            else:
                dl_args = dataloader_cfg.get("args", {})
            
            # Create DataLoader
            dataloader = DataLoader(dataset, **dl_args)
            dataloaders[dataset_name][split_name] = dataloader
            
            # Log dataloader creation
            batch_size = dl_args.get("batch_size", "unknown")
            num_workers = dl_args.get("num_workers", "unknown")
            logger.debug(f"Created {split_name} dataloader for {dataset_name}: "
                        f"batch_size={batch_size}, num_workers={num_workers}")
    
    return dataloaders