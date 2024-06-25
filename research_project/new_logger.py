from typing import Any, Dict, Optional

import os

import wandb
from loguru import logger

from research_project.utils import Config


class MLLogger:
    def __init__(
        self,
        cfg: Config,
    ):
        self.log_locally = cfg.log_locally
        self.log_file = cfg.log_file
        self.log_wandb = cfg.track
        self.wandb_project = cfg.wandb_project_name
        self.buffer = []
        self.learning_rate = None
        self.v_loss = None
        self.pg_loss = None
        self.entropy_loss = None
        self.old_approx_kl = None
        self.approx_kl = None
        self.clipfracs = None
        self.explained_var = None
        self.mean_episodic_return = None
        self.episode = None
        self.tax_frac = None
        self.mean_reward_across_envs = None
        self.principal_explained_var = None
        # Configure loguru logger
        logger.remove()  # Remove default handler
        if self.log_locally:
            logger.add(lambda msg: print(msg, end=""))
        if self.log_file:
            logger.add(self.log_file, rotation="10 MB")
        # Initialize wandb if needed
        if self.log_wandb:
            if self.wandb_project is None:
                raise ValueError(
                    "wandb_project must be provided when log_wandb is True"
                )
            wandb.init(project=self.wandb_project, config=cfg)

    def log(
        self,
        message: str,
        level: str = "INFO",
        wandb_data: Optional[Dict[str, Any]] = None,
        flush: bool = False,
    ):
        # Add to buffer
        self.buffer.append((message, level, wandb_data))
        if flush:
            self.flush()

    def flush(self):
        for message, level, wandb_data in self.buffer:
            # Log locally and/or to file
            if self.log_locally or self.log_file:
                getattr(logger, level.lower())(message)
            # Log to wandb
            if self.log_wandb:
                if wandb_data:
                    wandb.log(wandb_data)
                wandb.log({"message": message, "level": level})
        # Clear the buffer
        self.buffer.clear()

    def close(self):
        self.flush()
        if self.log_wandb:
            wandb.finish()


# ```
# This `MLLogger` class provides the following functionality:
# 1. It can log locally (to console), to a file, and to Weights & Biases (wandb), each independent of one another.
# 2. It has a single `.log()` method that supports buffered logging.
# 3. By default, `.log()` doesn't log immediately but stores messages in a buffer.
# 4. The `.flush()` method logs all buffered messages.
# 5. The `.log()` method also supports a `flush=True` argument to immediately log and flush the buffer.
# Here's how you can use this logger:
# ```python
# Initialize the logger
