"""Adders that use Reverb (github.com/deepmind/reverb) as a backend."""

import abc
import time
from typing import Callable, Iterable, Mapping, NamedTuple, Optional, Sized, Tuple, Union

import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree
from absl import logging

from marl import _types, worlds
from marl.services.replay.reverb.adders import base

DEFAULT_PRIORITY_TABLE = "priority_table"
_MIN_WRITER_LIFESPAN_SECONDS = 60


class Step(NamedTuple):
    """Step class used internally for reverb adders."""

    observation: _types.Tree
    action: _types.Tree
    reward: _types.Tree
    start_of_episode: Union[bool, worlds.ArraySpec, tf.Tensor, Tuple[()]]
    end_of_episode: Union[bool, worlds.ArraySpec, tf.Tensor, Tuple[()]]
    extras: _types.Tree = ()


class PriorityFnInput(NamedTuple):
    """The input to a priority function consisting of stacked steps."""

    observations: _types.Tree
    actions: _types.Tree
    rewards: _types.Tree
    start_of_episode: _types.Tree
    extras: _types.Tree


# Define the type of a priority function and the mapping from table to function.
PriorityFn = Callable[["PriorityFnInput"], float]
PriorityFnMapping = Mapping[str, Optional[PriorityFn]]


def spec_like_to_tensor_spec(paths: Iterable[str], spec: worlds.ArraySpec):
    """Convert a spec into a TF spec."""
    return tf.TensorSpec.from_spec(spec, name="/".join(str(p) for p in paths))


class ReverbAdder(base.Adder):
    """Base class for Reverb adders."""

    def __init__(
        self,
        client: reverb.Client,
        max_sequence_length: int,
        max_in_flight_items: int,
        delta_encoded: bool = False,
        priority_fns: Optional[PriorityFnMapping] = None,
        validate_items: bool = True,
    ):
        """Initialize a ReverbAdder instance.

        Args:
            client: A client to the Reverb backend.
            max_sequence_length: The maximum length of sequences (corresponding to the
                number of observations) that can be added to replay.
            max_in_flight_items: The maximum number of items allowed to be "in flight"
                at the same time. See `block_until_num_items` in
                `reverb.TrajectoryWriter.flush` for more info.
            delta_encoded: If `True` (False by default) enables delta encoding, see
                `Client` for more information.
            priority_fns: A mapping from table names to priority functions; if
                omitted, all transitions/steps/sequences are given uniform priorities
                (1.0) and placed in DEFAULT_PRIORITY_TABLE.
            validate_items: Whether to validate items against the table signature
                before they are sent to the server. This requires table signature to be
                fetched from the server and cached locally.
        """
        if priority_fns:
            priority_fns = dict(priority_fns)
        else:
            priority_fns = {DEFAULT_PRIORITY_TABLE: None}

        self._client = client
        self._priority_fns = priority_fns
        self._max_sequence_length = max_sequence_length
        self._delta_encoded = delta_encoded
        # TODO(b/206629159): Remove this.
        self._max_in_flight_items = max_in_flight_items

        # This is exposed as the _writer property in such a way that it will create
        # a new writer automatically whenever the internal __writer is None. Users
        # should ONLY ever interact with self._writer.
        self.__writer = None
        # Every time a new writer is created, it must fetch the signature from the
        # Reverb server. If this is set too low it can crash the adders in a
        # distributed setup where the replay may take a while to spin up.
        self._validate_items = validate_items

    def __del__(self):
        """Delete an instance of `ReverbAdder`."""
        if self.__writer is not None:
            timeout_ms = 10_000
            # Try flush all appended data before closing to avoid loss of experience.
            try:
                self.__writer.flush(0, timeout_ms=timeout_ms)
            except reverb.DeadlineExceededError as e:
                logging.error(
                    "Timeout (%d ms) exceeded when flushing the writer before "
                    "deleting it. Caught Reverb exception: %s",
                    timeout_ms,
                    str(e),
                )
            self.__writer.close()

    @property
    def _writer(self) -> reverb.TrajectoryWriter:
        """Get the underlying trajectory writer."""
        if self.__writer is None:
            self.__writer = self._client.trajectory_writer(
                num_keep_alive_refs=self._max_sequence_length, validate_items=self._validate_items
            )
            self._writer_created_timestamp = time.time()
        return self.__writer

    def add_priority_table(self, table_name: str, priority_fn: Optional[PriorityFn]):
        """Add a priority function for sampling from a table."""
        if table_name in self._priority_fns:
            raise ValueError(
                f"A priority function already exists for {table_name}. "
                f'Existing tables: {", ".join(self._priority_fns.keys())}.'
            )
        self._priority_fns[table_name] = priority_fn

    def reset(self, timeout_ms: Optional[int] = None):
        """Resets the adder's buffer."""
        if self.__writer:
            # Flush all appended data and clear the buffers.
            self.__writer.end_episode(clear_buffers=True, timeout_ms=timeout_ms)

            # Create a new writer unless the current one is too young.
            # This is to reduce the relative overhead of creating a new Reverb writer.
            if time.time() - self._writer_created_timestamp > _MIN_WRITER_LIFESPAN_SECONDS:
                self.__writer = None

    def add(self, timestep: dm_env.TimeStep, action: _types.Tree = None, extras: _types.Tree = ()):
        """Record an action and the following timestep."""
        if not timestep.first():
            # Complete the remaining row's information that was started during the previous timestep.
            self._writer.append(dict(reward=timestep.reward))
            self._write()

        has_extras = len(extras) > 0 if isinstance(extras, Sized) else extras is not None
        current_step = dict(
            observation=timestep.observation,
            start_of_episode=timestep.first(),
            end_of_episode=timestep.last(),
            action=action,
            **({"extras": extras} if has_extras else {}),
        )

        if timestep.first() or timestep.mid():
            # Start a new row based on the current observation. We write a partial row, because we are
            # awaiting the reward that is received for the action.
            self._writer.append(current_step, partial_step=True)

        elif timestep.last():
            # Place the final row which only contains the last observation for bootstrapping.
            # The remainder of the fields must be filled with dummy values for shape-correctness.
            dummy_step = tree.map_structure(np.zeros_like, current_step)
            dummy_step["observation"] = timestep.observation
            dummy_step["start_of_episode"] = timestep.first()
            dummy_step["end_of_episode"] = timestep.last()
            dummy_step["reward"] = tree.map_structure(np.zeros_like, timestep.reward)

            self._writer.append(dummy_step)
            self._write_last()
            self.reset()

        else:
            raise ValueError(f"Unknown timestep type: {timestep.step_type}.")

    @classmethod
    def signature(cls, environment_spec: worlds.EnvironmentSpec, extras_spec: worlds.TreeSpec = ()):
        """This is a helper method for generating signatures for Reverb tables.

        Signatures are useful for validating data types and shapes, see Reverb's
        documentation for details on how they are used.

        Args:
            environment_spec: A `specs.EnvironmentSpec` whose fields are nested
                structures with leaf nodes that have `.shape` and `.dtype` attributes.
                This should come from the environment that will be used to generate
                the data inserted into the Reverb table.
            extras_spec: A nested structure with leaf nodes that have `.shape` and
                `.dtype` attributes. The structure (and shapes/dtypes) of this must
                be the same as the `extras` passed into `ReverbAdder.add`.

        Returns:
            A `Step` whose leaf nodes are `tf.TensorSpec` objects.
        """
        spec_step = Step(
            observation=environment_spec.observations,
            action=environment_spec.actions,
            reward=environment_spec.rewards,
            start_of_episode=worlds.ArraySpec(shape=(), dtype=bool),
            end_of_episode=worlds.ArraySpec(shape=(), dtype=bool),
            extras=extras_spec,
        )
        return tree.map_structure_with_path(spec_like_to_tensor_spec, spec_step)

    @abc.abstractmethod
    def _write(self):
        """Write data to replay from the buffer."""

    @abc.abstractmethod
    def _write_last(self):
        """Write data to replay from the buffer."""
