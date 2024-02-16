"""Test for `learner_policy."""
from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from marl import types, worlds
from marl.services import learner_policy, test_utils
from marl.utils import tree_utils

_FAKE_TIMESTEP = worlds.TimeStep(
    step_type=worlds.StepType.FIRST,
    observation=1,
    reward=0,
)


class _TestPolicy(nn.Module):
  """Test policy that performs a linear operation.

  The policy computes m*x + b where the scale, m, is the parameter,
  and the bias, b, is the policy's state.

  https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.RNN.html
  https://flax.readthedocs.io/en/latest/guides/haiku_migration_guide.html
  """

  @nn.module.nowrap
  def step(self, state: types.State, timestep: worlds.TimeStep, rng_key):
    """Forward policy pass."""
    del rng_key
    self(timestep, state, None)

  @nn.module.nowrap
  def initialize_carry(self, rng: jax.random.PRNGKey, batch_shape: Tuple[int, ...]):
    del rng
    return jnp.ones(batch_shape[:-1], dtype=int)

  @nn.compact
  def __call__(self, state: types.State, timestep: worlds.TimeStep, rng_key):
    """Forward policy pass."""
    del rng_key
    observation = jnp.asarray(timestep.observation)
    mean = self.param(
        "mean",
        lambda _, shape: jnp.zeros(shape, dtype=int),  # rng, first param, is unused.
        observation.shape,
    )
    return state + 1, (observation * mean) + state


_POLICY = _TestPolicy()


class LearnPolicyTest(parameterized.TestCase):
  """Test suite for `LearnerPolicy`."""

  @parameterized.parameters(
      dict(
          expected_variables=[dict(params=dict(mean=i)) for i in range(10)],
          expected_actions=(2, 4, 6, 8, 10, 12, 14, 16, 18),
      ),
  )
  def test_step(
      self,
      expected_variables: Sequence[types.Tree],
      expected_actions: Sequence[types.Action],
  ):
    """Tests action selection of the policy.

    Args:
      expected_variables: List of the agent's variables that the agent is
        expected to pull from each call to StubVariableClient.
      expected_actions: Expected actions that an agent will take.
    """
    agent = learner_policy.LearnerPolicy(
        policy=_POLICY,
        variable_client=test_utils.StubVariableClient(expected_variables),
        rng_key=jax.random.PRNGKey(42),
        timestep_update_freq=1,
        episode_update_freq=None,
    )

    state = agent.episode_reset(_FAKE_TIMESTEP)
    for timestep_i in range(9):
      state, action = agent.step(state, _FAKE_TIMESTEP)
      tree_utils.assert_equals(
          action,
          expected_actions[timestep_i],
      )

  @parameterized.parameters(
      dict(
          expected_variables=[dict(params=dict(mean=i)) for i in range(10)],
          timestep_update_freq=1,
          include_resets=False,
      ),
      dict(
          expected_variables=[dict(params=dict(mean=i)) for i in range(10)],
          timestep_update_freq=1,
          include_resets=True,
      ),
      dict(
          expected_variables=[dict(params=dict(mean=i)) for i in range(10)],
          timestep_update_freq=3,
          include_resets=False,
      ),
      dict(
          expected_variables=[dict(params=dict(mean=i)) for i in range(10)],
          timestep_update_freq=3,
          include_resets=True,
      ),
  )
  def test_timestep_update(
      self,
      expected_variables: Sequence[types.Tree],
      timestep_update_freq: int,
      include_resets: bool,
  ):
    """Tests updating the agent's parameters at periodic timesteps.

    Args:
      expected_variables: List of the agent's variables that the agent is
        expected to pull from each call to StubVariableClient.
      timestep_update_freq: Frequency that the agent should update its
        variables by the number of timesteps received.
      include_resets: Reset the episode after each step. Used to ensure no
        cross-contamination with updating following episode counts.
    """
    agent = learner_policy.LearnerPolicy(
        policy=_POLICY,
        variable_client=test_utils.StubVariableClient(expected_variables),
        rng_key=jax.random.PRNGKey(42),
        timestep_update_freq=timestep_update_freq,
        episode_update_freq=None,
    )

    state = agent.episode_reset(_FAKE_TIMESTEP)
    for timestep_i in range(9):
      # Updates occur before an agent takes a step.
      tree_utils.assert_equals(
          agent.params,
          expected_variables[timestep_i // timestep_update_freq],
      )

      state, _ = agent.step(state, _FAKE_TIMESTEP)
      if include_resets:
        state = agent.episode_reset(_FAKE_TIMESTEP)

  @parameterized.parameters(
      dict(
          expected_variables=[dict(params=dict(mean=i)) for i in range(10)],
          episode_update_freq=1,
          include_steps=False,
      ),
      dict(
          expected_variables=[dict(params=dict(mean=i)) for i in range(10)],
          episode_update_freq=1,
          include_steps=True,
      ),
      dict(
          expected_variables=[dict(params=dict(mean=i)) for i in range(10)],
          episode_update_freq=3,
          include_steps=False,
      ),
      dict(
          expected_variables=[dict(params=dict(mean=i)) for i in range(10)],
          episode_update_freq=3,
          include_steps=True,
      ),
  )
  def test_episode_update(
      self,
      expected_variables: Sequence[types.Tree],
      episode_update_freq: int,
      include_steps: bool,
  ):
    """Tests updating the agent's parameters at periodic episodes.

    Args:
      expected_variables: List of the agent's variables that the agent is
        expected to pull from each call to StubVariableClient.
      episode_update_freq: Frequencye that the agent should update its
        variables by the number of episodes played.
      include_steps: Intermix agent `step` calls with episode resets. Used
        to ensure there is no cross-contamination with updating by timesteps.
    """
    agent = learner_policy.LearnerPolicy(
        policy=_POLICY,
        variable_client=test_utils.StubVariableClient(expected_variables),
        rng_key=jax.random.PRNGKey(42),
        timestep_update_freq=None,
        episode_update_freq=episode_update_freq,
    )

    for episode_i in range(9):
      # Updates occur before an agent resets episodic state.
      tree_utils.assert_equals(
          agent.params,
          expected_variables[episode_i // episode_update_freq],
      )

      state = agent.episode_reset(_FAKE_TIMESTEP)
      if include_steps:
        state, _ = agent.step(state, _FAKE_TIMESTEP)


if __name__ == "__main__":
  absltest.main()
