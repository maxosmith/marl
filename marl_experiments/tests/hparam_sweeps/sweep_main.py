import dataclasses
import os.path as osp
import subprocess
import tempfile

import ujson
from absl import app

from marl.utils import hyper
from marl_experiments.tests.hparam_sweeps import sweep_test


def main(_):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = sweep_test.Config()

        config_path = osp.join(tmp_dir, "config.json")
        ujson.dump(dataclasses.asdict(config), open(config_path, "w"))
        print(subprocess.getstatusoutput(f"python sweep_test.py --config={config_path}")[1])


if __name__ == "__main__":
    app.run(main)
