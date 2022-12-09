"""Perform sweeps over Roshambo experiments."""
import importlib
import os.path as osp
import subprocess
import time
from typing import Tuple

import ujson
from absl import app, flags, logging

from marl import utils
from marl.utils import config_dict_utils, hyper

# ========================================================================================
# Best Response
_BR_MAIN = "br_main"


def sweep_schedules() -> Tuple[hyper.Sweep, str]:
    """Sweep learning schedules."""
    sweep = hyper.product(
        [
            hyper.sweep("learning_rate_steps", [100_000, 50_000, 10_000, 4_000]),
            hyper.sweep("max_norm_steps", [100_000, 50_000, 10_000, 4_000]),
        ]
    )
    return sweep, _BR_MAIN


def sweep_baseline() -> Tuple[hyper.Sweep, str]:
    """Sweep over baseline BR(BR(Random))."""
    sweep = hyper.product(
        [
            hyper.fixed(
                "opponent_snapshot_path",
                "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_06_br_random_with_memory/wid_10/eval_arena/20221206-095632_59.79999923706055",
            ),
            hyper.sweep("optimizer.learning_rate_init", [6e-3, 6e-4, 6e-5, 6e-6, 6e-7, 6e-8]),
            hyper.fixed("optimizer.learning_rate_steps", None),
            hyper.sweep("optimizer.max_norm_init", [40.0, 10.0, 5.0, 1.0, 0.1]),
            hyper.fixed("optimizer.max_norm_steps", None),
            hyper.fixed("num_learner_steps", 5_000),
            hyper.sweep("agent.baseline_cost", [0.2]),
            hyper.sweep("agent.entropy_cost", [0.04]),
            hyper.fixed("agent.memory_core_ctor", "marl_experiments.gathering.networks.MemoryCore"),
        ]
    )
    return sweep, _BR_MAIN


# ========================================================================================
# World model training.
_TRAIN_WORLD_MODEL_MAIN = "train_world_model_main"


def sweep_reward() -> Tuple[hyper.Sweep, str]:
    """Sweeps the reward loss coefficient."""
    sweep = hyper.product([hyper.sweep("world_model.reward_cost", [1.0, 10.0, 25.0, 50.0, 75.0, 100.0, 1_000.0])])
    return sweep, _TRAIN_WORLD_MODEL_MAIN


# ========================================================================================
# Best Response with optional planning.
_BR_WITH_PLANNING_MAIN = "br_with_planning_main"


def sweep_planning() -> Tuple[hyper.Sweep, str]:
    """Hyperparameter sweep over only planning and not using a simulator.

    python br_with_planning_main.py \
        --test \
        --config.opponent_snapshot_path /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_06_br_random_with_memory/wid_10/eval_arena/20221206-095632_59.79999923706055 \
        --config.world_model_path /scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_06_train_world_model_memory/snapshot/20221206-162904/params
    """
    sweep = hyper.product(
        [
            hyper.fixed(
                "opponent_snapshot_path",
                "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_06_br_random_with_memory/wid_10/eval_arena/20221206-095632_59.79999923706055",
            ),
            hyper.fixed(
                "world_model_path",
                "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_06_train_world_model_memory/snapshot/20221206-162904/params",
            ),
            hyper.fixed("num_planner_steps", 5_000),
            hyper.fixed("num_learner_steps", 0),
            hyper.fixed("agent.baseline_cost", 0.2),
            hyper.fixed("agent.entropy_cost", 0.04),
            hyper.sweep("agent_optimizer.learning_rate_init", [6e-6, 6e-7, 6e-8, 6e-9]),
            hyper.sweep("agent_optimizer.max_norm_init", [40.0, 10.0, 5.0, 1.0, 0.1]),
        ]
    )

    return sweep, _BR_WITH_PLANNING_MAIN


def sweep_world_models() -> Tuple[hyper.Sweep, str]:
    """Sweeps over world models that vary in their reward's loss coefficient."""
    sweep = hyper.product(
        [
            hyper.fixed(
                "opponent_snapshot_path",
                "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_06_br_random_with_memory/wid_10/eval_arena/20221206-095632_59.79999923706055",
            ),
            hyper.sweep(
                "world_model_path",
                [
                    "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_08_world_model_reward_sweep_fixed/wid_0/snapshot/20221208-085701/params",
                    # "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_08_world_model_reward_sweep_fixed/wid_1/snapshot/20221208-085715/params",
                    # "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_08_world_model_reward_sweep_fixed/wid_2/snapshot/20221208-085714/params",
                    # "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_08_world_model_reward_sweep_fixed/wid_3/snapshot/20221208-085716/params",
                    # "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_08_world_model_reward_sweep_fixed/wid_4/snapshot/20221208-085714/params",
                    # "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_08_world_model_reward_sweep_fixed/wid_5/snapshot/20221208-085716/params",
                    # "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_08_world_model_reward_sweep_fixed/wid_6/snapshot/20221208-085714/params",
                ],
            ),
            # TODO(maxsmith): Switch to 5k.
            hyper.fixed("num_planner_steps", 4_000),
            hyper.fixed("num_learner_steps", 4_000),
            hyper.fixed("agent.baseline_cost", 0.2),
            hyper.fixed("agent.entropy_cost", 0.04),
            hyper.fixed("agent_optimizer.learning_rate_init", 6e-6),
            hyper.fixed("agent_optimizer.max_norm_init", 10.0),
        ]
    )

    return sweep, _BR_WITH_PLANNING_MAIN


# ========================================================================================

# Registers each sweep function to be accessible through a command line flag.
_SWEEPS = {
    "schedules": sweep_schedules,
    "baseline": sweep_baseline,
    "planning": sweep_planning,
    "reward": sweep_reward,
    "world_models": sweep_world_models,
}


_RESULT_DIR = "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/"
_EXPERIMENT_MODULE = "marl_experiments.one_step_transfer"

flags.DEFINE_string("name", None, "")
flags.DEFINE_enum("sweep", None, list(_SWEEPS.keys()), "")
FLAGS = flags.FLAGS


def main(_):
    """Runs experimental sweeps over `bc_main`."""
    result_dir = utils.ResultDirectory(osp.join(_RESULT_DIR, FLAGS.name), overwrite=True)

    sweep, experiment = _SWEEPS[FLAGS.sweep]()
    _check_for_unknown_keys(sweep, experiment)

    for wid, sweep in enumerate(sweep):
        # Create a subdirectory to store this worker's results.
        subdir = result_dir.make_subdir(f"wid_{wid}")
        # Write the worker's configuration into the result directory.
        ujson.dump(sweep.to_dict(), open(subdir.file("config.json"), "w"))

        # Build the SLURM launch command.
        command = f"sbatch --job-name {FLAGS.name}_{wid} "
        command += "scripts/cpu_short.sh "
        command += f"-r {subdir.dir}"
        command += ' "utils/sweep_worker.py '
        command += f"--main {_EXPERIMENT_MODULE}.{experiment} "
        command += f'--config_dir {subdir.dir}"'

        # Launch the worker on SLURM.
        output = subprocess.getstatusoutput(command)
        _, output = output
        _, _, _, job_id = output.split(" ")
        logging.info(f"WID {wid} launched on SLURM with JobID {job_id}.")

        # Politely wait to not overload the scheduler.
        time.sleep(0.5)


def _check_for_unknown_keys(sweep: Tuple[hyper.Sweep], experiment: str):
    """Checks if the sweep contains keys that are not defined in the experiment."""
    base_config = importlib.import_module(f"{_EXPERIMENT_MODULE}.{experiment}").get_config()
    unknown_keys = False

    # Check each worker's config generated by the sweep and tell the user of keys
    # that do not appear in the experiment's base config.
    for wid, worker_config in enumerate(sweep):
        keys = config_dict_utils.flatten_keys(worker_config)
        keys = [key for key in keys if not config_dict_utils.key_in(key, base_config)]
        if keys:
            logging.warning("Found keys that were not in the experiment's base config in WID %d: %s", wid, keys)
        unknown_keys = unknown_keys or keys

    # If we detect unknown keys ask the user to confirm that this is intentional.
    if unknown_keys:
        response = input("Proceed with launching jobs anyways [y]: ")
        if response != "y":
            logging.info("Not launching jobs.")
            exit(0)


if __name__ == "__main__":
    flags.mark_flag_as_required("name")
    flags.mark_flag_as_required("sweep")
    app.run(main)
