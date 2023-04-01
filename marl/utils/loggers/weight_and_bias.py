"""W&B Logger."""
from typing import Optional

import numpy as np
import wandb

from marl.utils.loggers import base


class WeightAndBiasLogger(base.Logger):
    """Logger that writes to W&B.

    NOTE: Currently only supports scalar logging.
    """

    def __init__(
        self,
        project_name: str,
        step_key: Optional[str] = None,
        run_name: Optional[str] = None,
        group_name: Optional[str] = None,
        group_logger_name: Optional[str] = None,
    ):
        """Initializes a `WeightAdnBiasLogger`.

        Args:
            project_name: Name of the experiment.
            step_key: Key in the log's data referring to the current step.
            run_name: Name of the run for an experiment.
                Use `group_name` if multiple processes are writing in the same run.
            group_name: Name of the run when it's distributed.
                Use `run_name` if it's a single-process run.
                `group_logger_name` must also be specified.
            group_logger_name: Name of the logger within the run's group.
                Used only when `group_name` is specified.
        """
        self._project_name = project_name
        self._step_key = step_key
        self._run_name = run_name
        self._group_name = group_name
        self._group_logger_name = group_logger_name

        if self._group_name and self._run_name:
            raise ValueError("Cannot specify both a group- and run-name.")

        if self._group_name and not self._group_logger_name:
            raise ValueError("Distributed logging requires `group_logger_name`.")

        if self._run_name:
            wandb.init(
                project=self._project_name,
                name=self._run_name,
            )
        else:
            wandb.init(
                project=self._project_name,
                group=self._group_name,
                job_type=self._group_logger_name,
            )

    def write(self, data: base.LogData):
        """Write log data."""
        step = data[self._step_key] if self._step_key is not None else None
        # TODO(maxsmith): Support more datatypes than scalars.
        formatted_data = {}
        for key, value in data.items():
            value = np.asarray(value)
            num_dims = len(value.shape)
            if num_dims == 0:
                formatted_data[key] = value

        wandb.log(data=formatted_data, step=step)

    def close(self):
        """Close W&B connection."""
        wandb.finish()
