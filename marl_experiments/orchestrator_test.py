import launchpad as lp
from absl import app, logging

from marl import services
from marl.utils import node_utils


@node_utils.build_courier_node
def build_counter():
    return services.Counter()


def main():
    program = lp.Program("test")

    controller = lp.launch(program)
    controller.wait()


if __name__ == "__main__":
    app.run(main)
