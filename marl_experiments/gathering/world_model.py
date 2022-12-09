from typing import Mapping, NamedTuple, Optional, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tree

from marl import _types, worlds
from marl.services.replay.reverb.adders import reverb_adder
from marl.services.replay.reverb.adders import utils as reverb_utils
from marl.utils import spec_utils


class WorldModelState(NamedTuple):
    recurrent_state: _types.Tree
    logits: _types.Logits
    prev_actions: _types.Tree


class WorldModel(hk.RNNCore):
    def __init__(
        self,
        num_players: int,
        state_shape: Tuple[int],
        *,
        # Sub-modules.
        input_encoder: hk.Module,
        memory_core: hk.Module,
        observation_prediction_head: hk.Module,
        reward_prediction_head: hk.Module,
        evaluation: bool,
        # Hyperparameters.
        reward_cost: float = 0.0,
        name: Optional[str] = "world_model",
    ) -> None:
        """Initialize an instance of the IMPALA algorithm's computational graphs."""
        super().__init__(name=name)
        self._num_players = num_players
        self._state_shape = state_shape
        self._reward_cost = reward_cost

        self._input_encoder = input_encoder
        self._memory_core = memory_core
        self._observation_prediction_heads = []
        self._reward_prediction_heads = []
        for _ in range(self._num_players):
            self._observation_prediction_heads.append(observation_prediction_head)
            self._reward_prediction_heads.append(reward_prediction_head)

        self._evaluation = evaluation

    def __call__(
        self, world_state: worlds.PlayerIDToTimestep, actions: _types.PlayerIDToAction, memory: WorldModelState
    ) -> worlds.PlayerIDToTimestep:
        """Forward transition dynamics of the world.

        Args:
            world_state: {P: [...]}.
            actions: {P: []}.
            model_state:

        Returns:
            Timesteps for each player.
        """
        # Process each agent's observation and previous-action.
        embeddings = []
        for player_id in range(self._num_players):
            embeddings.append(self._input_encoder(world_state[player_id].observation, actions[player_id]))
        embeddings = jnp.concatenate(embeddings, axis=-1)

        # Transition at the "world" level.
        embeddings, new_recurrent_state = self._memory_core(embeddings, memory.recurrent_state)

        # Make per-player predictions.
        timesteps = {}
        logits = {}
        for player_id in range(self._num_players):
            observation = self._observation_prediction_heads[player_id](embeddings)
            logits[player_id] = observation
            observation = jnp.argmax(observation, axis=-1)
            timesteps[player_id] = worlds.TimeStep(
                step_type=worlds.StepType.MID,
                observation=observation,
                reward=self._reward_prediction_heads[player_id](embeddings),
            )

        return timesteps, WorldModelState(recurrent_state=new_recurrent_state, logits=logits, prev_actions=actions)

    def initial_state(self, batch_size: Optional[int]):
        logits = np.zeros([batch_size, *self._state_shape] if batch_size else self._state_shape, dtype=np.float32)
        logits = {id: logits for id in range(self._num_players)}
        logits = {"world": logits}

        return WorldModelState(
            recurrent_state=self._memory_core.initial_state(batch_size),
            logits=logits,
            prev_actions=np.full([batch_size, 2] if batch_size else [2], -1, dtype=np.int32),
        )

    def state_spec(self) -> worlds.TreeSpec:
        return spec_utils.make_tree_spec(self.initial_state(None))

    def unroll(self, world_state: _types.Tree, actions: _types.Tree, recurrent_state: _types.Tree):
        """Efficient unroll that applies embeddings, MLP, & convnet in one pass.

        Args:
            world_state: per-player observations [B, T, P, ...].
            actions: per-player actions [B, T, P].
            recurrent_state: initial state for memory core with leaves of shape [B, ...].
        """
        # Process each agent's observation and previous-action.
        embeddings = []
        for player_id in range(self._num_players):
            embeddings.append(
                hk.BatchApply(self._input_encoder)(world_state[:, :, player_id], actions[:, :, player_id])
            )
        embeddings = jnp.concatenate(embeddings, axis=-1)

        # Roll forward the transition at the "world" level.
        embeddings, new_states = self._memory_core.unroll(embeddings, recurrent_state)

        # Make per-player predictions given the next "world" state.
        timesteps = {}
        logits = {}
        for player_id in range(self._num_players):
            observation = hk.BatchApply(self._observation_prediction_heads[player_id])(embeddings)
            logits[player_id] = observation
            observation = jnp.argmax(observation, axis=-1)
            timesteps[player_id] = worlds.TimeStep(
                step_type=worlds.StepType.MID,
                observation=observation,
                reward=hk.BatchApply(self._reward_prediction_heads[player_id])(embeddings),
            )

        return timesteps, WorldModelState(recurrent_state=new_states, logits=logits, prev_actions=actions[:, :-1])

    def loss(self, data: reverb_adder.Step) -> Tuple[_types.Tree, Mapping[str, _types.Tree]]:
        """Builds the loss function.

        Args:
            data: A batch of Sequence data, following the `Step` structure with leading
                dimensions: [B, T, P, ...].

        Returns:
            Loss and logging metrics.
        """
        observation, action, reward = (data.observation, data.action, data.reward)
        batch_size = observation.shape[0]
        initial_state = self.initial_state(batch_size)

        timesteps, extra = self.unroll(observation, action, initial_state.recurrent_state)

        pred_observations = {id: obs[:, :-1] for id, obs in extra.logits.items()}
        next_observation = {id: observation[:, 1:, id] for id in range(self._num_players)}
        reward = {id: reward[:, 1:, id] for id in range(self._num_players)}
        padding_mask = reverb_utils.padding_mask(data)[:, 1:]

        # Compute the observation's loss through cross entropy.
        observation_losses = {}
        for player_id, pred_observation in pred_observations.items():
            observation_loss = optax.softmax_cross_entropy(logits=pred_observation, labels=next_observation[player_id])
            observation_loss = jnp.mean(observation_loss, axis=[-2, -1])
            observation_loss = jnp.where(padding_mask, observation_loss, jnp.zeros_like(observation_loss))
            observation_losses[player_id] = jnp.mean(
                observation_loss
            )  # NOTE: This averaging does not consider padded arrays.

        # Compute the reward's loss through regression.
        reward_losses = {}
        for player_id, timestep in timesteps.items():
            reward_losses[player_id] = jnp.mean(optax.l2_loss(reward[player_id], timestep.reward[:, :-1]))

        total_observation_loss = jnp.mean(jnp.stack(list(observation_losses.values())))
        total_reward_loss = jnp.mean(jnp.stack(list(reward_losses.values())))
        loss = total_observation_loss + self._reward_cost * total_reward_loss

        metrics = dict(
            loss=loss,
            observation_loss=total_observation_loss,
            reward_loss=total_reward_loss,
            scaled_reward_loss=self._reward_cost * total_reward_loss,
        )
        for player_id, o_loss in observation_losses.items():
            metrics[f"observation_loss/{player_id}"] = o_loss
        for player_id, r_loss in reward_losses.items():
            metrics[f"reward_loss/{player_id}"] = r_loss

        metrics.update(self._accuracy_metrics(next_observation, pred_observations, padding_mask))
        return loss, metrics

    def _accuracy_metrics(
        self, next_observation: _types.Array, pred_observation: _types.Array, padding_mask: _types.Array
    ) -> Mapping[str, _types.Array]:
        """Compute accuracy metrics for a prediction.

        Args:
            next_observation: Ground-truth next observation [B, T, W, H, C].
            pred_observation: Predicted observation [B, T, W, H, C].
            padding_mask: Mask flagging if an example is padding [B, T].
        """
        metrics = {}

        # Convert from one-hot to class indices.
        next_classes, pred_classes = [], []
        for player_id in range(self._num_players):
            next_classes.append(jnp.argmax(next_observation[player_id], axis=-1))
            pred_classes.append(jnp.argmax(pred_observation[player_id], axis=-1))
        next_classes = jnp.concatenate(next_classes, axis=0)
        pred_classes = jnp.concatenate(pred_classes, axis=0)

        correct_predictions = next_classes == pred_classes

        # Reshape padding mask so that it can be applied across observations.
        padding_mask = padding_mask[:, :, None, None]  # [B, T] --> [B, T, W, H].
        padding_mask = jnp.concatenate(self._num_players * (padding_mask,), axis=0)  # [B*P, T, W, H].

        # Compute class-wise metrics (1 vs all).
        total_count = 0
        num_classes = next_observation[0].shape[-1]
        for i in range(num_classes):
            # The count of all of the actually positive examples: False Negative + True Positive.
            count = np.sum((next_classes == i) * padding_mask)
            total_count += count

            # Recall.
            true_positives = jnp.logical_and(correct_predictions, next_classes == i)
            true_positives = jnp.sum(true_positives * padding_mask)

            recall = true_positives / count
            metrics[f"recall/{i}"] = recall

            # Precision.
            false_negatives = jnp.logical_and(jnp.logical_not(correct_predictions), next_classes == i)
            false_negatives = jnp.sum(false_negatives * padding_mask)

            precision = true_positives / (true_positives + false_negatives)
            metrics[f"precision/{i}"] = precision

            # F1.
            metrics[f"f1/{i}"] = (2.0 * precision * recall) / (precision + recall)

        # Compute the overall accuracy.
        metrics["accuracy"] = jnp.sum(correct_predictions * padding_mask) / total_count
        return metrics
