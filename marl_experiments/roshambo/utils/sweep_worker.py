"""Runs a worker in a sweep.

It assumes that an experiment has two methods within its' driver:
    - get_config() -> ConfigDict: containing the default experiment configuration.
    - run(config): runs the experiment off of a configuration.
"""
import importlib
import os.path as osp

import ujson
from absl import app, flags

flags.DEFINE_string("main", None, "Module for the main program.")
flags.DEFINE_string("config_dir", None, "Result directory already containing this worker's config.")
FLAGS = flags.FLAGS


def main(_):
    """Build and run a single worker experiment."""
    config_path = osp.join(FLAGS.config_dir, "config.json")

    experiment_module = importlib.import_module(FLAGS.main)

    # Load the base config for the experiment and override it with sweep values.
    config = experiment_module.get_config()
    config.update(**ujson.load(open(config_path, "r")))

    experiment_module.run(config)


if __name__ == "__main__":
    flags.mark_flag_as_required("main")
    flags.mark_flag_as_required("config_dir")
    app.run(main)
