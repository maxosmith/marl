"""Runs a worker in a sweep."""
import os.path as osp

import ujson
from absl import app, flags

from marl_experiments.gathering import experiment

flags.DEFINE_string("config_dir", None, "")
FLAGS = flags.FLAGS


def main(_):
    config_path = osp.join(FLAGS.config_dir, "config.json")
    config = experiment.get_config()
    config.update(**ujson.load(open(config_path, "r")))
    print(config)
    experiment.run(config, exist_ok=True, overwrite=False)


if __name__ == "__main__":
    flags.mark_flag_as_required("config_dir")
    app.run(main)
