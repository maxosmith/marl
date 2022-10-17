import functools
from typing import List

import launchpad as lp
import reverb
from absl import app
from launchpad import context

from marl import worlds
from marl.services import arenas
from marl.rl.replay.reverb import adders
from marl.utils import mocks, signals, spec_utils


def build_reverb_node(env_spec: worlds.EnvironmentSpec) -> List[reverb.Table]:
    """Build the Reverb replay buffer."""
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=1_000_000,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=adders.NStepTransitionAdder.signature(env_spec),
    )
    return [replay_table]


def main(_):
    """Build and run the IMPALA distributed training topology."""
    game = mocks.DiscreteSymmetricSpecGame(num_actions=2, num_players=2)
    player_id_to_spec = spec_utils.make_game_specs(game)
    players = {id: mocks.Agent(spec) for id, spec in player_id_to_spec.items()}

    learner_id = 0
    env_spec = player_id_to_spec[learner_id]

    program = lp.Program(name="arena_test")

    with program.group("reverb"):
        make_reverb_node_fn = functools.partial(build_reverb_node, env_spec=env_spec)
        reverb_handle = program.add_node(lp.ReverbNode(make_reverb_node_fn))

    with program.group("training_arena"):
        train_node_handle = program.add_node(lp.CourierNode(arenas.TrainingArena, game, players))

    with program.group("evaluation_arena"):
        eval_node_handle = program.add_node(lp.CourierNode(arenas.EvaluationArena))

    with signals.runtime_terminator(lambda _: context.get_context().program_stopper()):
        lp.launch(program, launch_type=lp.LaunchType.TEST_MULTI_THREADING)

    eval_node_client = eval_node_handle.dereference()
    print(eval_node_client)

    x = eval_node_client.run_episode(game, players)
    print(x)

    x = eval_node_client.run_episodes(num_episodes=3, game=game, players=players)
    print(x)


if __name__ == "__main__":
    app.run(main)
