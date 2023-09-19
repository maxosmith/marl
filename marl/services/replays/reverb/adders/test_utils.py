"""Utilities for testing reverb adders."""
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import reverb
import tensorflow as tf
import tree
from absl.testing import absltest

from marl import specs, worlds
from marl.services.replays import end_behavior as end_behavior_lib
from marl.services.replays import priority
from marl.services.replays.reverb.adders import reverb_adder
from marl.utils import tree_utils

StepWithExtra = Tuple[Any, worlds.TimeStep, Any]
StepWithoutExtra = Tuple[Any, worlds.TimeStep]
Step = TypeVar("Step", StepWithExtra, StepWithoutExtra)


def restart(observation):
    """Returns a `TimeStep` with `step_type` set to `StepType.FIRST`."""
    return worlds.TimeStep(worlds.StepType.FIRST, None, observation)


def transition(reward, observation):
    """Returns a `TimeStep` with `step_type` set to `StepType.MID`."""
    return worlds.TimeStep(worlds.StepType.MID, reward, observation)


def termination(reward, observation):
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`."""
    return worlds.TimeStep(worlds.StepType.LAST, reward, observation)


def truncation(reward, observation):
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`."""
    return worlds.TimeStep(worlds.StepType.LAST, reward, observation)


def _numeric_to_spec(x: Union[float, int, np.ndarray]):
    if isinstance(x, np.ndarray):
        return specs.ArraySpec(shape=x.shape, dtype=x.dtype)
    elif isinstance(x, (float, int)):
        return specs.ArraySpec(shape=(), dtype=type(x))
    else:
        raise ValueError(f"Unsupported numeric: {type(x)}")


def get_specs(step):
    """Infer spec from an example step."""
    env_spec = tree.map_structure(
        _numeric_to_spec,
        specs.EnvironmentSpec(
            observation=step[1].observation,
            action=step[0],
            reward=step[1].reward if step[1].reward else 0.0,
        ),
    )

    has_extras = len(step) == 3
    if has_extras:
        extras_spec = tree.map_structure(_numeric_to_spec, step[2])
    else:
        extras_spec = ()

    return env_spec, extras_spec


class ReverbAdderTestMixin(absltest.TestCase):
    """A helper mixin for testing Reverb adders.

    Note that any test inheriting from this mixin must also inherit from something
    that provides the Python unittest assert methods.
    """

    server: reverb.Server
    client: reverb.Client

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        replay_table = reverb.Table.queue(priority.DEFAULT_PRIORITY_TABLE, 1000)
        cls.server = reverb.Server([replay_table])
        cls.client = reverb.Client(f"localhost:{cls.server.port}")

    def tearDown(self):
        self.client.reset(priority.DEFAULT_PRIORITY_TABLE)
        super().tearDown()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        super().tearDownClass()

    def num_episodes(self):
        info = self.client.server_info(1)[priority.DEFAULT_PRIORITY_TABLE]
        return info.num_episodes

    def num_items(self):
        info = self.client.server_info(1)[priority.DEFAULT_PRIORITY_TABLE]
        return info.current_size

    def items(self):
        sampler = self.client.sample(
            table=priority.DEFAULT_PRIORITY_TABLE,
            num_samples=self.num_items(),
            emit_timesteps=False,
        )
        return [sample.data for sample in sampler]  # pytype: disable=attribute-error

    def run_test_adder(
        self,
        adder: reverb_adder.ReverbAdder,
        steps: Sequence[Step],
        expected_items: Sequence[Any],
        signature: specs.TreeSpec,
        stack_sequence_fields: bool = True,
        repeat_episode_times: int = 1,
        end_behavior: end_behavior_lib.EndBehavior = end_behavior_lib.EndBehavior.ZERO_PAD,
        item_transform: Optional[Callable[[Sequence[np.ndarray]], Any]] = None,
    ):
        """Runs a unit test case for the adder.

        Args:
          adder: The instance of `Adder` that is being tested.
          steps: A sequence of (action, timestep) tuples that are passed to
            `Adder.add()`.
          expected_items: The sequence of items that are expected to be created
            by calling the adder's `add_first()` method on `first` and `add()` on
            all of the elements in `steps`.
          signature: Signature that written items must be compatible with.
          stack_sequence_fields: Whether to stack the sequence fields of the
            expected items before comparing to the observed items. Usually False
            for transition adders and True for both episode and sequence adders.
          repeat_episode_times: How many times to run an episode.
          end_behavior: How end of episode should be handled.
          item_transform: Transformation of item simulating the work done by the
            dataset pipeline on the learner in a real setup.
        """

        if not steps:
            raise ValueError("At least one step must be given.")

        has_extras = len(steps[0]) == 3
        for _ in range(repeat_episode_times):
            for step in steps:
                ts, action = step[1], step[0]
                extras = step[2] if has_extras else ()
                adder.add(timestep=ts, action=action, extras=extras)

        # Force run the destructor to trigger the flushing of all pending items.
        getattr(adder, "__del__", lambda: None)()

        # Ending the episode should close the writer. No new writer should yet have
        # been created as it is constructed lazily.
        if end_behavior is not end_behavior_lib.EndBehavior.CONTINUE:
            self.assertEqual(self.num_episodes(), repeat_episode_times)

        # Make sure our expected and observed data match.
        observed_items = self.items()

        # Check matching number of items.
        self.assertEqual(len(expected_items), len(observed_items))

        # Check items are matching according to numpy's almost_equal.
        for expected_item, observed_item in zip(expected_items, observed_items):
            if stack_sequence_fields:
                expected_item = tree_utils.stack(expected_item)

            # Apply the transformation which would be done by the dataset in a real
            # setup.
            if item_transform:
                observed_item = item_transform(observed_item)

            tree.map_structure(
                np.testing.assert_array_almost_equal,
                tree.flatten(expected_item),
                tree.flatten(observed_item),
            )

        # Make sure the signature matches was is being written by Reverb.
        def _check_signature(spec: tf.TensorSpec, value: np.ndarray):
            self.assertTrue(spec.is_compatible_with(tf.convert_to_tensor(value)))

        # Check that it is possible to unpack observed using the signature.
        for item in observed_items:
            tree.map_structure(_check_signature, tree.flatten(signature), tree.flatten(item))
