from typing import Any, Dict, Optional

import os

import wandb
from loguru import logger

from harvest_sed.utils import Config


class MLLogger:
    def __init__(
        self,
        cfg: Config,
    ):
        self.log_locally = cfg.log_locally
        self.log_file = cfg.log_file
        self.log_wandb = cfg.track
        self.wandb_project = cfg.wandb_project_name
        self.wandb_entity = cfg.wandb_entity
        self.buffer = []
        self.return_buffer = []
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
            log_directory = os.path.dirname(cfg.log_file)
            if not os.path.exists(log_directory):
                os.makedirs(log_directory)
            logger.add(self.log_file, rotation="10 MB")
            print(
                os.path.abspath(cfg.log_file)
            )  # This will show the absolute path used by the logger

        # Initialize wandb if needed
        if self.log_wandb:
            if self.wandb_project is None:
                raise ValueError(
                    "wandb_project must be provided when log_wandb is True"
                )
            wandb.init(project=self.wandb_project, entity=self.wandb_entity, config=cfg)
            wandb.define_metric("opt/epoch")
            wandb.define_metric("principal_opt/epoch")
            wandb.define_metric("opt/*", step_metric="opt/epoch")
            wandb.define_metric("principal_opt/*", step_metric="prinicpal_opt/epoch")

            wandb.define_metric("principal_train_avg/step")
            wandb.define_metric("train_avg/step")
            wandb.define_metric(
                "principal_train_avg/*", step_metric="principal_train_avg/step"
            )
            wandb.define_metric("train_avg/*", step_metric="train_avg/step")

            wandb.define_metric("principal_episode_eval/step")
            wandb.define_metric("episode_eval/step")
            wandb.define_metric(
                "principal_episode_eval/*", step_metric="principal_episode_eval/step"
            )
            wandb.define_metric("episode_eval/*", step_metric="episode_eval/step")

            wandb.define_metric("collect/step")
            wandb.define_metric("/*", step_metric="collect/step")

    def log(
        self,
        wandb_data: Optional[Dict[str, Any]] = None,
        flush: bool = False,
    ):
        # Add to buffer
        self.buffer.append(wandb_data)
        if flush:
            self.flush()

    def log_return(
        self,
        wandb_data: Optional[Dict[str, Any]] = None,
    ):
        self.return_buffer.append(wandb_data)

    def flush_return(self):
        for idx in range(len(self.buffer)):
            self.buffer[idx].update(self.return_buffer[idx])
            wandb.log(self.buffer[idx])
        self.return_buffer.clear()

    def flush(self):
        for wandb_data in self.buffer:
            # Log locally and/or to file
            # if self.log_locally or self.log_file:
            #     getattr(logger, level.lower())(message)
            # Log to wandb
            wandb.log(wandb_data)
        self.buffer.clear()

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
