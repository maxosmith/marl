"""Perform sweeps over Roshambo experiments."""
import os.path as osp
import subprocess
import time
from typing import Tuple

import ujson
from absl import app, flags, logging

from marl import utils
from marl.utils import hyper
from marl_experiments.roshambo import roshambo_bot

# ========================================================================================
# Best Response
_TRAIN_RESPONSE_MAIN = "train_response_main"


def sweep_br_ps() -> Tuple[hyper.Sweep, str]:
    """Computes best-responses to each pure-strategy bot."""
    sweep = hyper.product(
        [
            hyper.sweep("opponents", [[name] for name in roshambo_bot.ROSHAMBO_BOT_NAMES]),
        ]
    )
    return sweep, _TRAIN_RESPONSE_MAIN


def sweep_br_ms() -> Tuple[hyper.Sweep, str]:
    """Compute a best-response to the uniform mixted strategy opponent."""
    sweep = hyper.product([hyper.sweep("opponents", [roshambo_bot.ROSHAMBO_BOT_NAMES])])
    return sweep, _TRAIN_RESPONSE_MAIN


# ========================================================================================
# Behavioural Cloning
_BC_MAIN = "bc_main"


def sweep_bc_ps() -> Tuple[hyper.Sweep, str]:
    """Perform behavioural cloning on for each agent, playing against the population."""
    sweep = hyper.product(
        [
            hyper.sweep("bot_names", [[name] for name in roshambo_bot.ROSHAMBO_BOT_NAMES]),
        ]
    )
    return sweep, _BC_MAIN


def sweep_bc_ms() -> Tuple[hyper.Sweep, str]:
    """Perform behavioural cloning on for each agent, playing against the population."""
    sweep = hyper.product(
        [
            hyper.sweep("bot_names", [roshambo_bot.ROSHAMBO_BOT_NAMES]),
        ]
    )
    return sweep, _BC_MAIN


# ========================================================================================

# Registers each sweep function to be accessible through a command line flag.
_SWEEPS = {
    "br_ps": sweep_br_ps,
    "br_ms": sweep_br_ms,
    "bc_ps": sweep_bc_ps,
    "bc_ms": sweep_bc_ms,
}


_RESULT_DIR = "/scratch/wellman_root/wellman1/mxsmith/results/marl/roshambo/"

flags.DEFINE_string("name", None, "")
flags.DEFINE_enum("sweep", None, list(_SWEEPS.keys()), "")
FLAGS = flags.FLAGS


def main(_):
    """Runs experimental sweeps over `bc_main`."""
    result_dir = utils.ResultDirectory(osp.join(_RESULT_DIR, FLAGS.name), overwrite=True)

    sweep, experiment = _SWEEPS[FLAGS.sweep]()

    for wid, sweep in enumerate(sweep):
        # Create a subdirectory to store this worker's results.
        subdir = result_dir.make_subdir(f"wid_{wid}")
        sweep["result_dir"] = subdir.dir
        # Write the worker's configuration into the result directory.
        ujson.dump(sweep.to_dict(), open(subdir.file("config.json"), "w"))

        # Build the SLURM launch command.
        command = f"sbatch --job-name {FLAGS.name}_{wid} "
        command += f'scripts/cpu_day.sh "utils/sweep_worker.py '
        command += f"--main marl_experiments.roshambo.{experiment} "
        command += f'--config_dir {subdir.dir}"'

        # Launch the worker on SLURM.
        output = subprocess.getstatusoutput(command)
        _, output = output
        _, _, _, job_id = output.split(" ")
        logging.info(f"WID {wid} launched on SLURM with JobID {job_id}.")

        # Politely wait to not overload the scheduler.
        time.sleep(0.5)


if __name__ == "__main__":
    flags.mark_flag_as_required("name")
    flags.mark_flag_as_required("sweep")
    app.run(main)
