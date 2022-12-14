"""Mock classes used for testing."""
from typing import Optional, Sequence

import dm_env
import numpy as np

from marl import worlds
from marl.utils import spec_utils


class Environment(worlds.Environment):
    """A fake environment with a given spec."""

    def __init__(
        self,
        spec: worlds.EnvironmentSpec,
        *,
        episode_length: int = 25,
    ):
        self._spec = spec
        self._episode_length = episode_length
        self._step = 0

    def _generate_fake_observation(self):
        return spec_utils.generate_from_spec(self._spec.observation)

    def _generate_fake_reward(self):
        return spec_utils.generate_from_spec(self._spec.reward)

    def reset(self) -> worlds.TimeStep:
        observation = self._generate_fake_observation()
        self._step = 1
        return dm_env.restart(observation)

    def step(self, action) -> worlds.TimeStep:
        # Return a reset timestep if we haven't touched the environment yet.
        if not self._step:
            return self.reset()

        spec_utils.validate_spec(self._spec.action, action)

        observation = self._generate_fake_observation()
        reward = self._generate_fake_reward()

        if self._episode_length and (self._step == self._episode_length):
            self._step = 0
            # We can't use dm_env.termination directly because then the discount
            # wouldn't necessarily conform to the spec (if eg. we want float32).
            return worlds.TimeStep(worlds.StepType.LAST, reward, observation)
        else:
            self._step += 1
            return dm_env.transition(reward=reward, observation=observation)

    def action_spec(self):
        return self._spec.action

    def observation_spec(self):
        return self._spec.observation

    def reward_spec(self):
        return self._spec.reward


class _BaseDiscreteEnvironment(Environment):
    """Discrete action fake environment."""

    def __init__(
        self,
        *,
        num_actions: int = 1,
        action_dtype=np.int32,
        observation_spec: worlds.TreeSpec,
        reward_spec: Optional[worlds.TreeSpec] = None,
        **kwargs,
    ):
        """Initialize the environment."""
        if reward_spec is None:
            reward_spec = worlds.ArraySpec((), np.float32)

        actions = worlds.DiscreteArraySpec(num_actions, dtype=action_dtype)

        super().__init__(
            spec=worlds.EnvironmentSpec(
                observation=observation_spec,
                action=actions,
                reward=reward_spec,
            ),
            **kwargs,
        )


class DiscreteEnvironment(_BaseDiscreteEnvironment):
    """Discrete state and action fake environment."""

    def __init__(
        self,
        *,
        num_actions: int = 1,
        action_dtype=np.int32,
        obs_dtype=np.int32,
        obs_shape: Sequence[int] = (),
        reward_spec: Optional[worlds.TreeSpec] = None,
        **kwargs,
    ):
        """Initialize the environment."""
        observations_spec = worlds.ArraySpec(shape=obs_shape, dtype=obs_dtype)

        super().__init__(
            num_actions=num_actions,
            action_dtype=action_dtype,
            observation_spec=observations_spec,
            reward_spec=reward_spec,
            **kwargs,
        )
