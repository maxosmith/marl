"""Utilities for building a launch-pad program for distributed off-policy training.

References:
    - https://github.com/deepmind/acme/acme/jax/experiments/make_distributed_experiment.py
    - https://github.com/ethanluoyc/magi/blob/main/magi/layouts/distributed_layout.py
"""
from typing import Callable, Optional, Sequence, Union

import haiku as hk
import jax
import launchpad as lp

from marl import utils

_CourierCtor = Callable[..., lp.CourierNode]
_ReverbCtor = Callable[..., lp.ReverbNode]


def build_distributed_training_program(
    learner_ctor: _CourierCtor,
    train_arena_ctor: _CourierCtor,
    eval_arena_ctors: Union[_CourierCtor, Sequence[_CourierCtor]],
    replay_ctor: _ReverbCtor,
    counter_ctor: _CourierCtor,
    result_dir: utils.ResultDirectory,
    num_train_arenas: int = 1,
    seed: int = 42,
    name: Optional[str] = None,
    program: Optional[lp.Program] = None,
    collocate_eval_arenas: bool = False,
) -> lp.Program:
    """Builds a distributed off-policy training program.

    This layout currently expects that only one individual is learning within any given game.

    Layouts are only responsible for handling program-level node arguments such as
    the random key, result directories, and handles for other nodes.

    Args:
        random_key:
        name:
        program:
    """
    # Either specify a program or name for the new program.
    if (program is None) and (name is None):
        raise ValueError("Please specify either an existing program to modify, or a name for a new program.")
    if (name is not None) and (program is not None):
        raise ValueError("Cannot specify provide both an existing program and a name for a new program.")

    program = program if program else lp.Program(name=name)
    key_sequence = hk.PRNGSequence(seed)
    eval_arena_ctors = eval_arena_ctors if isinstance(eval_arena_ctors, Sequence) else [eval_arena_ctors]

    with program.group("replay"):
        replay_handle = program.add_node(replay_ctor())

    with program.group("counter"):
        counter_handle = program.add_node(counter_ctor())

    with program.group("learner"):
        learner_handle = program.add_node(
            learner_ctor(
                replay=replay_handle,
                counter=counter_handle,
                random_key=next(key_sequence),
                result_dir=result_dir.make_subdir(program._current_group),
            )
        )

    # TODO(maxsmith): Create several variable sources (e.g., through lp.CacherNode), so that
    # training arenas do not have to all share the same variable source.
    with program.group("train_arena"):
        for i in range(num_train_arenas):
            program.add_node(
                train_arena_ctor(
                    learner=learner_handle,
                    counter=counter_handle,
                    random_key=next(key_sequence),
                    result_dir=result_dir.make_subdir(f"{program._current_group}_{i}"),
                )
            )

    with program.group("eval_arena"):
        # Build all of evaluation nodes.
        eval_nodes = []
        for i, eval_arena_ctor in enumerate(eval_arena_ctors):
            eval_nodes.append(
                eval_arena_ctor(
                    learner=learner_handle,
                    counter=counter_handle,
                    random_key=next(key_sequence),
                    result_dir=result_dir.make_subdir(f"{program._current_group}_{i}"),
                )
            )

        # Optionally collocate all of the nodes.
        if collocate_eval_arenas:
            program.add_node(lp.MultiThreadingColocation(eval_nodes))
        else:
            _ = [program.add_node(node) for node in eval_nodes]

    return program
