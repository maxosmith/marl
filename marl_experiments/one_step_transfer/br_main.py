from typing import Optional

import haiku as hk
import launchpad as lp
import ujson
from absl import app, flags, logging
from ml_collections import config_dict, config_flags

from marl import bots as bots_lib
from marl import services, utils
from marl.services import snapshotter
from marl.utils import spec_utils
from marl_experiments.one_step_transfer import configs as exp_configs
from marl_experiments.one_step_transfer.utils import builders, graphs


def get_config() -> config_dict.ConfigDict:
    """Default configuration for this experiment."""
    config = config_dict.create(
        seed=42,
        step_key="learner_step",
        frame_key="learner_frame",
        opponent_snapshot_path=None,
        # opponent_snapshot_path="/scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/br_main/snapshot/20221129-144818/impala/",
        num_learner_steps=5_000,
        num_train_arenas=4,
        agent=exp_configs.agent(),
        replay=exp_configs.agent_replay(),
        # Optimizer configuration.
        optimizer=exp_configs.agent_optimizer(),
        # Evaluation.
        eval_frequency=100,
    )
    return config


_CONFIG = config_flags.DEFINE_config_dict(
    name="config",
    config=get_config(),
)
flags.DEFINE_bool("test", False, "Reduces the number of learner steps.")
flags.DEFINE_string(
    "result_dir",
    "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/br_main/",
    "Result directory.",
    short_name="r",
)
FLAGS = flags.FLAGS


def run(config: Optional[config_dict.ConfigDict] = None, exist_ok: bool = False, overwrite: bool = True):
    """Build and then run (see `Scheduler`) an one-step transfer experiment."""
    config = config if config else _CONFIG.value
    if FLAGS.test:
        logging.info("Overriding config with test settings.")
        config = config.unlock()
        config.num_learner_steps = 4
        config = config.lock()
    logging.info("Experiment's configuration:\n %s", config)

    result_dir = config.result_dir if "result_dir" in config else FLAGS.result_dir
    result_dir = utils.ResultDirectory(result_dir, exist_ok=exist_ok, overwrite=overwrite)
    ujson.dump(config.to_dict(), open(result_dir.file("config.json"), "w"))

    key_sequence = hk.PRNGSequence(config.seed)
    program = lp.Program(name="experiment")

    game = builders.build_game()
    env_spec = spec_utils.make_game_specs(game)[0]
    agent_graphs = graphs.build_agent_graphs(config.agent, env_spec)

    if config.opponent_snapshot_path:
        bots = {
            1: services.EvaluationPolicy(
                policy_fn=agent_graphs.eval_policy,
                initial_state_fn=agent_graphs.initial_state,
                random_key=next(key_sequence),
                params=snapshotter.restore_from_path(config.opponent_snapshot_path).params,
            )
        }
    else:
        bots = {1: bots_lib.RandomIntAction(num_actions=8)}

    snapshot_template = services.Snapshot(
        ctor=graphs.build_agent_graphs,
        ctor_kwargs=dict(config=config, env_spec=env_spec),
    )

    # ==============================================================================================
    # Build the program.

    with program.group("replay"):
        replay = program.add_node(
            builders.build_agent_replay(
                replay_config=config.replay,
                env_spec=spec_utils.make_game_specs(game)[0],
                state_and_extra_spec=agent_graphs.state_spec,
            )
        )

    with program.group("counter"):
        counter = program.add_node(builders.build_counter())

    with program.group("learner"):
        learner_handle = program.add_node(
            builders.build_agent_learner(
                result_dir=result_dir.make_subdir(program._current_group),
                random_key=next(key_sequence),
                step_key=config.step_key,
                frame_key=config.frame_key,
                optimizer_config=config.optimizer,
                replay_config=config.replay,
                loss_graph=agent_graphs.loss,
                replay=replay,
                counter=counter,
            )
        )

    with program.group("train_arena"):
        train_arena_handles = []
        for i in range(config.num_train_arenas):
            train_arena_handles.append(
                program.add_node(
                    builders.build_agent_training_arena(
                        result_dir=result_dir.make_subdir(f"{program._current_group}_{i}"),
                        random_key=next(key_sequence),
                        step_key=config.step_key,
                        policy_graph=agent_graphs.policy,
                        initial_state_graph=agent_graphs.initial_state,
                        learner=learner_handle,
                        counter=counter,
                        replay=replay,
                        replay_config=config.replay,
                        game=game,
                        bots=bots,
                    )
                )
            )

    with program.group("eval_arena"):
        eval_arena_handle = program.add_node(
            builders.build_evaluation_arena_node(
                result_dir=result_dir.make_subdir(program._current_group),
                random_key=next(key_sequence),
                step_key=config.step_key,
                eval_frequency=config.eval_frequency,
                policy_graph=agent_graphs.eval_policy,
                initial_state_graph=agent_graphs.initial_state,
                learner=learner_handle,
                bots=bots,
                counter=counter,
                game_ctor=builders.build_game,
                snapshot_template=snapshot_template,
            )
        )

    with program.group("snapshot"):
        program.add_node(
            builders.build_snapshot_node(
                snapshot_template=snapshot_template,
                learner_update_handle=learner_handle,
                result_dir=result_dir.make_subdir(program._current_group),
            )
        )

    with program.group("stopper"):
        stopper_handle = program.add_node(builders.build_stopper())

    # ==============================================================================================

    logging.info("Launching program.")
    manager = lp.launch(program, launch_type=lp.LaunchType.LOCAL_MULTI_PROCESSING, terminal="current_terminal")

    logging.info("Starting arenas.")
    arena_clients = [handle.dereference() for handle in train_arena_handles + [eval_arena_handle]]
    for arena_client in arena_clients:
        arena_client.futures.run()

    logging.info("Starting learner.")
    learner_handle.dereference().run(config.num_learner_steps)

    logging.info("Stopping arenas.")
    for arena_client in arena_clients:
        arena_client.stop()

    logging.info("Experiment finished.")
    stopper_handle.dereference().stop()
    manager.wait()


def main(_):
    """Enables running the file directly through absl, and also running with a config input."""
    run()


if __name__ == "__main__":
    app.run(main)
