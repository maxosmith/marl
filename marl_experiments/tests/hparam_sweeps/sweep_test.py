import dataclasses

import ujson
from absl import app, flags

flags.DEFINE_string("config", default="", help="No.")
FLAGS = flags.FLAGS


@dataclasses.dataclass
class Config:
    x: float = 0.003


def main(_):
    print(FLAGS.config)
    print(type(FLAGS.config))
    config = Config(**ujson.load(open(FLAGS.config, "r")))
    print(config)


if __name__ == "__main__":
    app.run(main)
