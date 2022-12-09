"""Compute BR by first planning and then acting."""
import os
from typing import Optional

import haiku as hk
import launchpad as lp
import ujson
from absl import app, flags, logging
from ml_collections import config_dict, config_flags

from marl import bots as bots_lib
from marl import services, utils
from marl.services import snapshotter
from marl.utils import spec_utils, wrappers
from marl_experiments.gathering import world_model_game_proxy
from marl_experiments.one_step_transfer import configs as exp_configs
from marl_experiments.one_step_transfer.utils import builders, graphs


def get_config() -> config_dict.ConfigDict:
    """Default configuration for this experiment."""
    config = config_dict.create(
        seed=42,
        sequence_length=20,
        sequence_period=None,
        step_key="learner_step",
        frame_key="learner_frame",
        opponent_snapshot_path="",
        # opponent_snapshot_path="/scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/br_main/snapshot/20221129-144818/impala/",
        world_model_path="/scratch/wellman_root/wellman1/mxsmith/results/marl/gathering/one_step_transfer/"
        "train_world_model/snapshot/20221130-151502/params/",
        num_planner_steps=5_000,
        num_learner_steps=5_000,
        num_train_arenas=4,
        num_plan_arenas=4,
        # Evaluation.
        eval_frequency=100,
        # Components.
        agent=exp_configs.agent(),
        agent_replay=exp_configs.agent_replay(),
        agent_optimizer=exp_configs.agent_optimizer(),
        world_model=exp_configs.world_model(),
    )
    return config


_CONFIG = config_flags.DEFINE_config_dict(
    name="config",
    config=get_config(),
)
flags.DEFINE_bool("test", False, "Reduces the number of learner steps.")
flags.DEFINE_string(
    "result_dir",
    "/scratch/wellman_root/wellman1/mxsmith/results/marl/one_step_transfer/br_with_planning_main/",
    "Result directory.",
    short_name="r",
)
FLAGS = flags.FLAGS


def run(config: Optional[config_dict.ConfigDict] = None, exist_ok: bool = False, overwrite: bool = True):
    """Build and then run an one-step transfer experiment."""
    config = config if config else _CONFIG.value
    if FLAGS.test:
        logging.info("Overriding config with test settings.")
        config = config.unlock()
        config.num_planner_steps = 4
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

    # Set-up an opponent policy.
    if config.opponent_snapshot_path:
        logging.info("Loading coplayer from snapshot: %s", config.opponent_snapshot_path)
        bots = {
            1: services.EvaluationPolicy(
                policy_fn=agent_graphs.eval_policy,
                initial_state_fn=agent_graphs.initial_state,
                random_key=next(key_sequence),
                params=snapshotter.restore_from_path(config.opponent_snapshot_path).params,
            )
        }
    else:
        logging.info("Using a random coplayer.")
        bots = {1: bots_lib.RandomIntAction(num_actions=8)}

    # Load a previously trained world model.
    logging.info("Loading world model from snapshot: %s", config.world_model_path)
    world_model_graphs = graphs.build_world_graphs(config.world_model, env_spec)
    world_model_params = snapshotter.restore_from_path(config.world_model_path).params[0]

    snapshot_template = services.Snapshot(
        ctor=graphs.build_agent_graphs,
        ctor_kwargs=dict(config=config, env_spec=env_spec),
    )

    # ==============================================================================================
    # Build the program.

    with program.group("replay"):
        replay_handle = program.add_node(
            builders.build_agent_replay(
                env_spec=spec_utils.make_game_specs(game)[0],
                state_and_extra_spec=agent_graphs.state_spec,
                replay_config=config.agent_replay,
            )
        )

    with program.group("counter"):
        counter_handle = program.add_node(builders.build_counter())

    with program.group("learner"):
        learner_handle = program.add_node(
            builders.build_agent_learner(
                result_dir=result_dir.make_subdir(program._current_group),
                random_key=next(key_sequence),
                step_key=config.step_key,
                frame_key=config.frame_key,
                optimizer_config=config.agent_optimizer,
                replay_config=config.agent_replay,
                loss_graph=agent_graphs.loss,
                replay=replay_handle,
                counter=counter_handle,
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
                        counter=counter_handle,
                        replay=replay_handle,
                        replay_config=config.agent_replay,
                        game=game,
                        bots=bots,
                    )
                )
            )

    with program.group("plan_arena"):
        plan_arena_handles = []
        for i in range(config.num_plan_arenas):
            plan_arena_handles.append(
                program.add_node(
                    builders.build_agent_planning_arena_without_var_servers(
                        result_dir=result_dir.make_subdir(f"{program._current_group}_{i}"),
                        step_key=config.step_key,
                        random_key=next(key_sequence),
                        policy_graph=agent_graphs.policy,
                        initial_state_graph=agent_graphs.initial_state,
                        learner=learner_handle,
                        counter=counter_handle,
                        replay=replay_handle,
                        replay_config=config.agent_replay,
                        game=game,
                        players=bots,
                        transition_graph=world_model_graphs.transition,
                        world_init_graph=world_model_graphs.initial_state,
                        world_model_params=world_model_params,
                    )
                )
            )

    # Evaluation arenas are currently seperate for Planning/Training stages to make pulling logs easier.
    with program.group("plan_eval_real_arena"):
        plan_eval_real_arena_handle = program.add_node(
            builders.build_evaluation_arena_node(
                step_key=config.step_key,
                eval_frequency=config.eval_frequency,
                random_key=next(key_sequence),
                policy_graph=agent_graphs.eval_policy,
                initial_state_graph=agent_graphs.initial_state,
                learner=learner_handle,
                bots=bots,
                counter=counter_handle,
                game_ctor=builders.build_game,
                snapshot_template=snapshot_template,
                result_dir=result_dir.make_subdir(program._current_group),
            )
        )

    with program.group("train_eval_real_arena"):
        train_eval_real_arena_handle = program.add_node(
            builders.build_evaluation_arena_node(
                step_key=config.step_key,
                eval_frequency=config.eval_frequency,
                random_key=next(key_sequence),
                policy_graph=agent_graphs.eval_policy,
                initial_state_graph=agent_graphs.initial_state,
                learner=learner_handle,
                bots=bots,
                counter=counter_handle,
                game_ctor=builders.build_game,
                snapshot_template=snapshot_template,
                result_dir=result_dir.make_subdir(program._current_group),
            )
        )

    learned_game = world_model_game_proxy.WorldModelGameProxy(
        transition=world_model_graphs.transition,
        init_state=world_model_graphs.initial_state,
        params=world_model_params,
        random_key=next(key_sequence),
        game=game,
    )
    learned_game = wrappers.TimeLimit(learned_game, num_steps=100)

    with program.group("plan_eval_learned_arena"):
        plan_eval_learned_arena_handle = program.add_node(
            builders.build_evaluation_arena_node(
                step_key=config.step_key,
                eval_frequency=config.eval_frequency,
                random_key=next(key_sequence),
                policy_graph=agent_graphs.eval_policy,
                initial_state_graph=agent_graphs.initial_state,
                learner=learner_handle,
                bots=bots,
                counter=counter_handle,
                game_ctor=lambda: learned_game,
                snapshot_template=snapshot_template,
                result_dir=result_dir.make_subdir(program._current_group),
            )
        )

    with program.group("train_eval_learned_arena"):
        train_eval_learned_arena_handle = program.add_node(
            builders.build_evaluation_arena_node(
                step_key=config.step_key,
                eval_frequency=config.eval_frequency,
                random_key=next(key_sequence),
                policy_graph=agent_graphs.eval_policy,
                initial_state_graph=agent_graphs.initial_state,
                learner=learner_handle,
                bots=bots,
                counter=counter_handle,
                game_ctor=lambda: learned_game,
                snapshot_template=snapshot_template,
                result_dir=result_dir.make_subdir(program._current_group),
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
    terminal = "output_to_files" if "LAUNCHPAD_LOGGING_DIR" in os.environ else "current_terminal"
    manager = lp.launch(program, launch_type=lp.LaunchType.LOCAL_MULTI_PROCESSING, terminal=terminal)
    learner_client = learner_handle.dereference()

    logging.info("Warm-starting through planning.")
    plan_arena_clients = [handle.dereference() for handle in plan_arena_handles]
    plan_arena_clients.append(plan_eval_real_arena_handle.dereference())
    plan_arena_clients.append(plan_eval_learned_arena_handle.dereference())
    for plan_client in plan_arena_clients:
        plan_client.futures.run()

    learner_client.run(config.num_planner_steps)

    for plan_client in plan_arena_clients:
        plan_client.stop()

    logging.info("Fine-tuning through acting.")
    train_arena_clients = [handle.dereference() for handle in train_arena_handles]
    train_arena_clients.append(train_eval_real_arena_handle.dereference())
    train_arena_clients.append(train_eval_learned_arena_handle.dereference())
    for arena_client in train_arena_clients:
        arena_client.futures.run()

    learner_client.run(config.num_learner_steps)

    for train_client in train_arena_clients:
        train_client.stop()

    logging.info("Experiment finished.")
    stopper_handle.dereference().stop()
    manager.wait()


def main(_):
    """Enables running the file directly through absl, and also running with a config input."""
    run()


if __name__ == "__main__":
    app.run(main)
