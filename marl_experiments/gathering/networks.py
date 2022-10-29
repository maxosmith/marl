from typing import Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from marl import _types, nets, worlds
from marl.rl.agents.impala.impala import IMPALAState


class CNNTimestepEncoder(hk.Module):
    def __init__(self, num_actions: int, name: Optional[str] = "timestep_encoder"):
        super().__init__(name=name)
        self.num_actions = num_actions
        self._observation_net = hk.Sequential(
            [
                # Input: [h, w, 4].
                hk.ConvND(num_spatial_dims=2, output_channels=4, kernel_shape=5, stride=1),
                jax.nn.relu,
                hk.ConvND(num_spatial_dims=2, output_channels=2, kernel_shape=5, stride=1),
                jax.nn.relu,
                hk.Flatten(),
            ]
        )
        self._net = hk.nets.MLP([20, 20])

    def __call__(self, timestep: worlds.TimeStep, state: IMPALAState) -> _types.Tree:
        observation = timestep.observation.astype(float)
        # ConvND assumes there is a leading batch dimension: [B, H, W, C].
        if len(observation.shape) == 3:
            # Add a dummy batch dimension, this typically needs to happen during graph building.
            observation = jnp.expand_dims(observation, axis=0)
            h = self._observation_net(observation)
            # Remove the dummy batch dimension.
            h = jnp.squeeze(h, axis=0)
        else:
            h = self._observation_net(observation)
        action = jax.nn.one_hot(state.prev_action, self.num_actions)
        h = jnp.concatenate([h, action], axis=-1)
        return self._net(h)


class MLPTimestepEncoder(hk.Module):
    def __init__(self, num_actions: int, name: Optional[str] = "timestep_encoder"):
        super().__init__(name=name)
        self.num_actions = num_actions
        self._observation_net = hk.nets.MLP([512, 256], activate_final=True)
        self._net = hk.nets.MLP([256, 256], activate_final=True)

    def __call__(self, timestep: worlds.TimeStep, state: IMPALAState) -> _types.Tree:
        observation = timestep.observation.astype(float)
        # Flatten assumes there is a leading batch dimension: [B, H, W, C].
        observation = jnp.ravel(observation) if len(observation.shape) == 3 else hk.Flatten()(observation)
        h = self._observation_net(observation)
        action = jax.nn.one_hot(state.prev_action, num_classes=self.num_actions, axis=-1)
        h = jnp.concatenate([h, action], axis=-1)
        return self._net(h)


class MemoryCore(hk.Module):
    def __init__(self, name: Optional[str] = "memory_core"):
        super().__init__(name=name)
        self._core = hk.LSTM(256)

    def __call__(self, inputs: _types.Tree, state: hk.LSTMState) -> Tuple[_types.Tree, hk.LSTMState]:
        outputs, new_state = self._core(inputs, state)
        return outputs, new_state

    def initial_state(self, batch_size: Optional[int]) -> hk.LSTMState:
        return self._core.initial_state(batch_size)

    def unroll(self, inputs: _types.Tree, state: hk.LSTMState) -> Tuple[_types.Tree, hk.LSTMState]:
        """This should be for additional time dimension over call"""
        outputs, new_state = hk.static_unroll(
            core=self._core, input_sequence=inputs, initial_state=state, time_major=False
        )
        return outputs, new_state


class MemoryLessCore(hk.Module):
    def __init__(self, name: Optional[str] = "memoryless_core"):
        super().__init__(name=name)
        self._core = nets.MLPCore([256])

    def __call__(self, inputs: _types.Tree, state: nets.MLPCoreState) -> Tuple[_types.Tree, nets.MLPCoreState]:
        outputs, new_state = self._core(inputs, state)
        return outputs, new_state

    def initial_state(self, batch_size: Optional[int]) -> nets.MLPCoreState:
        return self._core.initial_state(batch_size)

    def unroll(self, inputs: _types.Tree, state: nets.MLPCoreState) -> Tuple[_types.Tree, nets.MLPCoreState]:
        """This should be for additional time dimension over call"""
        outputs, new_state = hk.static_unroll(
            core=self._core, input_sequence=inputs, initial_state=state, time_major=False
        )
        return outputs, new_state


class PolicyHead(hk.Module):
    def __init__(self, num_actions: int, name: Optional[str] = "policy_head"):
        super().__init__(name=name)
        self.num_actions = num_actions
        self._policy_head = hk.Linear(self.num_actions)

    def __call__(self, inputs: _types.Tree) -> Tuple[_types.Action, _types.Tree]:
        logits = self._policy_head(inputs)  # [B, A]
        return logits


class ValueHead(hk.Module):
    def __init__(self, name: Optional[str] = "value_head"):
        super().__init__(name=name)
        self._value_head = hk.Linear(1)

    def __call__(self, inputs: _types.Tree) -> Tuple[_types.Action, _types.Tree]:
        value = jnp.squeeze(self._value_head(inputs), axis=-1)  # [B]
        return value


class WorldStateConvPredictionHead(hk.Module):
    def __init__(self, state_shape, name: Optional[str] = "world_state_prediction_head"):
        super().__init__(name=name)

    def __call__(self, x: _types.Tree) -> _types.Tree:
        x = hk.Linear(13 * 11 * 2)(x)
        x = jax.nn.relu(x)
        x = jnp.reshape(x, x.shape[:-1] + (13, 11, 2))
        x = hk.ConvNDTranspose(
            num_spatial_dims=2,
            output_channels=2,
            kernel_shape=5,
            stride=1,
        )(x)
        x = jax.nn.relu(x)
        x = hk.ConvNDTranspose(
            num_spatial_dims=2,
            output_channels=4,
            kernel_shape=5,
            stride=1,
        )(x)
        return x  # (..., 21, 19, 4).


class WorldStateLinearEncoder(hk.Module):
    """World model input encoder for global-state with joint-actions."""

    def __init__(self, state_shape: Tuple[int], num_actions: int, name: Optional[str] = "input_encoder"):
        super().__init__(name=name)
        self._state_shape = state_shape
        self._num_actions = num_actions
        final_output_shape = np.prod(self._state_shape)
        self._net = hk.nets.MLP([final_output_shape, int(final_output_shape / 2), int(final_output_shape / 2)])

    def __call__(self, world_state: _types.Tree, actions: _types.PlayerIDToAction) -> _types.Tree:
        world_state = hk.Flatten(preserve_dims=-3)(world_state)
        actions = hk.Flatten(preserve_dims=-2)(jax.nn.one_hot(actions, self._num_actions))
        inputs = jnp.concatenate([world_state, actions], axis=-1)
        return self._net(inputs)


class WorldStateLinearPredictionHead(hk.Module):
    def __init__(self, state_shape: Tuple[int], name: Optional[str] = "world_state_prediction_head"):
        super().__init__(name=name)
        self._state_shape = state_shape
        final_output_shape = np.prod(self._state_shape)
        self._net = hk.nets.MLP([int(final_output_shape / 2), int(final_output_shape / 2), final_output_shape])

    def __call__(self, x: _types.Tree) -> _types.Tree:
        x = self._net(x)
        x = jnp.reshape(x, x.shape[:-1] + self._state_shape)
        return x


class RewardPredictionHead(hk.Module):
    def __init__(self, name: Optional[str] = "reward_prediction_head"):
        super().__init__(name=name)
        self._net = hk.nets.MLP([256, 128, 32, 1])

    def __call__(self, x: _types.Tree) -> _types.Tree:
        return self._net(x)
