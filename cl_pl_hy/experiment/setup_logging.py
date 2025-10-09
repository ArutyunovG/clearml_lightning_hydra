import logging
from omegaconf import DictConfig

def setup_logging(cfg: DictConfig) -> None:
    """Setup logging configuration based on the config."""
    logger = logging.getLogger('cpplhy.experiment.setup_logging')

    # Configure logging based on config, with INFO as default
    log_level_str = cfg.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Reconfigure logging with the config-specified level
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True  # Override existing configuration
    )
    logger.info(f'Log level reset to: {log_level_str} based on config')
