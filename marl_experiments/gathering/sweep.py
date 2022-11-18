import dataclasses
import os.path as osp
import subprocess
import time

import ujson
from absl import app, flags, logging

from marl import utils
from marl.utils import hyper
from marl_experiments.gathering import experiment

_RESULT_DIR = "/scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/"

flags.DEFINE_string("name", None, "")
FLAGS = flags.FLAGS


def get_sweep() -> hyper.Sweep:
    return hyper.product(
        [
            hyper.sweep("optimizer_name", ["adam", "rmsprop"]),
            hyper.sweep("optimizer_config.learning_rate_end", [6e-8, 6e-9, 3e-9, 6e-10]),
            hyper.sweep("optimizer_config.max_norm_end", [0.1, 1.0, 5.0, 10.0]),
        ]
    )


def main(_):
    config = experiment.get_config()
    sweeps = hyper.default(config, get_sweep())
    result_dir = utils.ResultDirectory(osp.join(_RESULT_DIR, FLAGS.name), overwrite=True)

    for wid, sweep in enumerate(sweeps):
        # Set-up a sweep-worker by first building its result directory and providing it with it's config.
        subdir = result_dir.make_subdir(f"wid_{wid}")
        sweep["result_dir"] = subdir.dir
        ujson.dump(sweep.to_dict(), open(subdir.file("config.json"), "w"))

        # Each sweep-worker is then launched as an independent SLURM job, and respectively loads it's config
        # within its driver.
        output = subprocess.getstatusoutput(
            f"sbatch --job-name {FLAGS.name}_{wid} scripts/sweep_worker.sh {subdir.dir}"
        )
        _, output = output
        _, _, _, job_id = output.split(" ")
        logging.info(f"WID {wid} launched on SLURM with JobID {job_id}.")

        # Politely wait to not DDOS the scheduler.
        time.sleep(0.1)


if __name__ == "__main__":
    flags.mark_flag_as_required("name")
    app.run(main)
