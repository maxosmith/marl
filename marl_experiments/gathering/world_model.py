from typing import Mapping, NamedTuple, Optional, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tree

from marl import _types, worlds
from marl.rl.replay.reverb.adders import reverb_adder
from marl.rl.replay.reverb.adders import utils as reverb_utils
from marl.utils import spec_utils


class WorldModelState(NamedTuple):
    recurrent_state: _types.Tree
    logits: _types.Logits
    prev_actions: _types.Tree


class WorldModel(hk.RNNCore):
    def __init__(
        self,
        state_shape: Tuple[int],
        *,
        # Sub-modules.
        input_encoder: hk.Module,
        memory_core: hk.Module,
        state_prediction_head: hk.Module,
        reward_prediction_head: hk.Module,
        evaluation: bool,
        # Hyperparameters.
        reward_cost: float = 0.0,
        name: Optional[str] = "world_model",
    ) -> None:
        """Initialize an instance of the IMPALA algorithm's computational graphs."""
        super().__init__(name=name)
        self._state_shape = state_shape
        self._reward_cost = reward_cost

        self._input_encoder = input_encoder
        self._memory_core = memory_core
        self._state_prediction_head = state_prediction_head
        self._reward_prediction_head = reward_prediction_head
        self._evaluation = evaluation

    def __call__(
        self, world_state: _types.Tree, actions: _types.PlayerIDToAction, state: WorldModelState
    ) -> _types.Action:
        embeddings = self._input_encoder(world_state, actions)
        embeddings, new_recurrent_state = self._memory_core(embeddings, state.recurrent_state)
        world_state_logits = self._state_prediction_head(embeddings)
        world_state = jnp.argmax(world_state_logits, axis=-1)
        reward_logits = self._reward_prediction_head(embeddings)
        reward = jnp.argmax(reward_logits, axis=-1)
        return (world_state, reward), WorldModelState(
            recurrent_state=new_recurrent_state,
            logits={"world": world_state_logits, "reward": reward_logits},
            prev_actions=actions,
        )

    def initial_state(self, batch_size: Optional[int]):
        return WorldModelState(
            recurrent_state=self._memory_core.initial_state(batch_size),
            logits={
                "world": np.zeros(
                    [batch_size, *self._state_shape] if batch_size else self._state_shape, dtype=np.float32
                ),
                "reward": np.zeros([batch_size] if batch_size else [], dtype=np.float32),
            },
            prev_actions=np.full([batch_size, 2] if batch_size else [2], -1, dtype=np.int32),
        )

    def state_spec(self) -> worlds.TreeSpec:
        return spec_utils.make_tree_spec(self.initial_state(None))

    def unroll(self, world_state: _types.Tree, actions: _types.PlayerIDToAction, recurrent_state: _types.Tree):
        """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
        embeddings = hk.BatchApply(self._input_encoder)(world_state, actions)
        embeddings, new_states = self._memory_core.unroll(embeddings, recurrent_state)
        world_state_logits = hk.BatchApply(self._state_prediction_head)(embeddings)
        world_state = jnp.argmax(world_state_logits, axis=-1)
        reward_logits = hk.BatchApply(self._reward_prediction_head)(embeddings)
        reward = jnp.argmax(reward_logits, axis=-1)
        return (world_state, reward), WorldModelState(
            recurrent_state=new_states,
            logits={"world": world_state_logits, "reward": reward_logits},
            prev_actions=actions[:, :-1],
        )

    def loss(self, data: reverb_adder.Step) -> Tuple[_types.Tree, Mapping[str, _types.Tree]]:
        """Builds the loss function.

        Args:
            data: A batch of Sequence data, following the `Step` structure with leading
                dimensions: [B, T, ...].

        Returns:
            Loss and logging metrics.
        """
        observation, action, reward, state_and_extras = (
            data.observation,
            data.action,
            data.reward,
            data.extras,
        )
        batch_size = observation.shape[0]
        initial_state = self.initial_state(batch_size)

        _, extra = self.unroll(observation, state_and_extras, initial_state.recurrent_state)
        pred_observation, pred_reward = extra.logits["world"], extra.logits["reward"]

        # We do not have a label for the last observation's prediction.
        pred_observation = pred_observation[:, :-1]
        next_observation = observation[:, 1:]
        pred_reward = pred_reward[:, :-1]
        reward = reward[:, 1:]
        padding_mask = reverb_utils.padding_mask(data)[:, 1:]

        observation_loss = distrax.Categorical(probs=next_observation).cross_entropy(
            distrax.Categorical(logits=pred_observation)
        )
        observation_loss = jnp.nan_to_num(observation_loss)  # Mask out NaNs from padded observations.
        observation_loss = (observation_loss.T * padding_mask.T).T  # Put [B, T] at end for broadcasting.
        observation_loss = jnp.mean(observation_loss)
        # reward_loss = distrax.Categorical(probs=reward).cross_entropy(distrax.Categorical(logits=pred_reward))
        reward_loss = 0.0
        reward_loss = jnp.mean(reward_loss)

        loss = observation_loss + self._reward_cost * reward_loss
        metrics = dict(
            loss=loss,
            observation_loss=observation_loss,
            reward_loss=reward_loss,
            scaled_reward_loss=self._reward_cost * reward_loss,
        )
        return loss, metrics
