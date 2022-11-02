import dataclasses
import os.path as osp
import subprocess

import ujson
from absl import app, flags, logging

from marl import utils
from marl.utils import hyper
from marl_experiments.gathering import experiment

flags.DEFINE_string("result_dir", "/scratch/wellman_root/wellman1/mxsmith/tests/impala_sweep2", "")
FLAGS = flags.FLAGS


def get_sweep() -> hyper.Sweep:
    return hyper.product(
        [
            hyper.sweep("batch_size", [32, 64]),
            hyper.sweep("learning_rate", [3e-3, 6e-3, 3e-4, 6e-4]),
            hyper.sweep("baseline_cost", [0.5, 0.25, 0.1]),
            hyper.sweep("entropy_cost", [0.01, 0.02]),
        ]
    )


def main(_):
    config = experiment.Config()
    sweeps = hyper.default(config, get_sweep())
    result_dir = utils.ResultDirectory(FLAGS.result_dir, overwrite=True)

    for wid, sweep in enumerate(sweeps):
        subdir = result_dir.make_subdir(f"wid_{wid}")
        sweep["result_dir"] = subdir.dir
        ujson.dump(sweep, open(subdir.file("config.json"), "w"))
        output = subprocess.getstatusoutput(f"sbatch launch.sh {subdir.dir}")
        _, output = output
        _, _, _, job_id = output.split(" ")
        logging.info(f"WID {wid} launched on SLURM with JobID {job_id}.")


if __name__ == "__main__":
    app.run(main)
