from typing import Optional

import haiku as hk
import launchpad as lp
import numpy as np
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
        num_learner_steps=10_000,
        # Previously trained bot configs.
        bot_snapshot_paths=[
            "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_12_06_br_random_with_memory/wid_10/eval_arena/20221206-095632_59.79999923706055",
            # "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/2022_11_29_br_sweep/wid_10/snapshot/20221129-163717/impala/",
            None,
        ],
        world_model=exp_configs.world_model(),
        replay=exp_configs.world_replay(),
        optimizer=exp_configs.world_optimizer(),
        agent=exp_configs.agent(),
    )
    return config


_CONFIG = config_flags.DEFINE_config_dict(
    name="config",
    config=get_config(),
)
flags.DEFINE_bool("test", False, "Reduces amount of data generated and learner steps.")
flags.DEFINE_string(
    "result_dir",
    "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/train_world_model/",
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
        config.replay.replay_max_size = 10_000
        config.num_learner_steps = 4
        config = config.lock()
    logging.info("Experiment's configuration:\n %s", config)

    result_dir = utils.ResultDirectory(FLAGS.result_dir, exist_ok, overwrite=overwrite)
    ujson.dump(config.to_dict(), open(result_dir.file("config.json"), "w"))

    key_sequence = hk.PRNGSequence(config.seed)
    program = lp.Program(name="experiment")

    game = builders.build_game()
    env_spec = spec_utils.make_game_specs(game)[0]
    agent_graphs = graphs.build_agent_graphs(config.agent, env_spec)

    bots = {
        0: bots_lib.RandomIntAction(num_actions=8),
        1: bots_lib.RandomIntAction(num_actions=8),
    }

    if config.bot_snapshot_paths:
        logging.info("Replacing random policies with frozen trained policies.")
        for player_id, path in enumerate(config.bot_snapshot_paths):
            # Allow the user to specify random policies with None.
            if path is None:
                continue
            logging.info("\t-%d: %s", player_id, path)
            bots[player_id] = services.EvaluationPolicy(
                policy_fn=agent_graphs.eval_policy,
                initial_state_fn=agent_graphs.initial_state,
                random_key=next(key_sequence),
                params=snapshotter.restore_from_path(path).params,
            )

    world_model = graphs.build_world_graphs(config.world_model, env_spec)

    # ==============================================================================================
    # Build the program.

    with program.group("reverb"):
        reverb_handle = program.add_node(
            builders.build_world_replay(config.replay, spec_utils.make_game_specs(game)[0])
        )

    with program.group("counter"):
        counter_handle = program.add_node(builders.build_counter())

    with program.group("arena"):
        arena_handle = program.add_node(builders.build_world_train_arena(config.replay, game, bots, reverb_handle))

    with program.group("learner"):
        learner_handle = program.add_node(
            builders.build_world_model_learner(
                config.step_key,
                config.frame_key,
                next(key_sequence),
                world_model.loss,
                counter_handle,
                reverb_handle,
                result_dir.make_subdir(program._current_group),
                replay_config=config.replay,
                optimizer_config=config.optimizer,
            )
        )

    # TODO(maxsmith): This interface assumes that we have one pretrained policy.
    with program.group("render"):
        render_handle = program.add_node(
            builders.build_render_arena_node(
                step_key=config.step_key,
                bots=bots,
                result_dir=result_dir.make_subdir(program._current_group),
            )
        )

    with program.group("snapshot"):
        snapshot_template = services.Snapshot(
            ctor=graphs.build_world_graphs,
            ctor_kwargs=dict(config=config, env_spec=env_spec),
        )
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
    lp.launch(program, launch_type=lp.LaunchType.LOCAL_MULTI_PROCESSING, terminal="current_terminal")

    logging.info("Rendering sample of dataset data...")
    render_handle.dereference().run_evaluation(step=0)

    logging.info("Generating dataset...")
    arena_handle.dereference().run(num_timesteps=config.replay.replay_max_size)

    logging.info("Beginning learning...")
    learner_handle.dereference().run(num_steps=config.num_learner_steps)

    logging.info("Experiment finished.")
    stopper_handle.dereference().stop()


def main(_):
    """Enables running the file directly through absl, and also running with a config input."""
    run()


if __name__ == "__main__":
    app.run(main)
