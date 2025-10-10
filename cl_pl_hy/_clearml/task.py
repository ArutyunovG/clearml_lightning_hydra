# c_pl_hy/_clearml/task.py
from __future__ import annotations
from typing import Optional
import logging

from clearml import Task
from omegaconf import DictConfig


logger = logging.getLogger("cpplhy._clearml.task")


class ClearMLTask:
    """Wrapper around clearml.Task that accepts clearml subconfig and handles setup."""

    def __init__(
        self,
        clearml_config: DictConfig
    ) -> None:
        """
        Initialize ClearML Task from clearml configuration.
        
        Args:
            clearml_config: The clearml section from the main config
        """
        # Extract basic task info
        project_name = clearml_config.get("project_name", "default_project")
        task_name = clearml_config.get("task_name", "default_task")
        
        logger.info("Initializing ClearML Task: %s/%s", project_name, task_name)
        
        # Initialize the task
        self.task = Task.init(
            project_name=project_name, 
            task_name=task_name
        )
        
        # Handle pip requirements for remote execution
        self._setup_pip_requirements(clearml_config)
        self._setup_docker_config(clearml_config)
        
        logger.info("ClearML Task initialized successfully. Task ID: %s", self.task.id)
    

    def _setup_pip_requirements(self, config: DictConfig) -> None:
        """Setup pip requirements from config."""
        pip_config = config.get("pip")
        if pip_config and "requirements" in pip_config:
            requirements = list(pip_config.requirements)
            logger.info(f"Adding {requirements} pip requirements to task")
            self.task.set_packages(list(requirements))
        else:
            logger.debug("No pip requirements found in config")
    

    def _setup_docker_config(self, config: DictConfig) -> None:
        """Setup docker configuration from config."""
        docker_config = config.get("docker")
        if docker_config:
            image = docker_config.get("image")
            args = docker_config.get("args", [])
            env = docker_config.get("env", [])
            
            logger.info("Setting up Docker configuration: %s", image)
            
            if image:
                # Set the docker image for remote execution
                self.task.set_base_docker(docker_image=image)
                logger.info("Docker image set: %s", image)
            
            if args:
                # Set docker arguments
                self.task.set_base_docker(docker_arguments=args)
                logger.info("Docker arguments set: %s", args)
            
            if env:
                # Convert environment list to dict format if needed
                env_dict = {}
                for env_var in env:
                    if "=" in env_var:
                        key, value = env_var.split("=", 1)
                        env_dict[key] = value
                    else:
                        # Handle env vars without values
                        env_dict[env_var] = ""
                
                # Set environment variables for docker
                for key, value in env_dict.items():
                    self.task.set_parameter(f"docker_env/{key}", value)
                
                logger.info("Docker environment variables set: %s", list(env_dict.keys()))
        else:
            logger.debug("No docker configuration found in config")

    # Convenience passthrough if needed later
    def __getattr__(self, item):
        return getattr(self.task, item)
