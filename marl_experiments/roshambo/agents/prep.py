import functools
from typing import Mapping, NamedTuple, Optional, Tuple

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import tree

from marl import _types, worlds
from marl.services.replay.reverb.adders import reverb_adder
from marl.services.replay.reverb.adders import utils as reverb_utils
from marl.utils import loggers, spec_utils


class PREPState(NamedTuple):
    recurrent_state: _types.Tree
    logits: _types.Logits
    prev_action: _types.Action


class PREP(hk.RNNCore):
    def __init__(
        self,
        *,
        # Sub-modules.
        conditional_policy: hk.Module,
        evidence_encoder: hk.Module,
        id_embedder: hk.Module,
        best_responder: hk.Module,
        # Parameters.
        num_actions: int,
        # Hyperparameters.
        discount: float = 0.99,
        max_abs_reward: float = np.inf,
        baseline_cost: float = 1.0,
        entropy_cost: float = 0.0,
        name: Optional[str] = "impala",
    ):
        """Initialize an instance of the IMPALA algorithm's computational graphs.

        Args:
            timestep_encoder: Module that learns an embedding for a timestep.
            memory_core: Module responsible for maintaining state/memory.
            policy_head: Module that predicts per-action logits.
            value_head: Module that predicts state values.
            evaluation: Whether action selection should occur with exploration (True) or without (False).
        """
        super().__init__(name=name)
        self._num_actions = num_actions

        self._discount = discount
        self._max_abs_reward = max_abs_reward
        self._baseline_cost = baseline_cost
        self._entropy_cost = entropy_cost

        self._conditional_policy = conditional_policy
        self._evidence_encoder = evidence_encoder
        self._id_embedder = id_embedder
        self._best_responder = best_responder

    def __call__(self, timestep: worlds.TimeStep, state: PREPState) -> Tuple[_types.Action, PREPState]:
        """Forward pass of IMPALA's policy."""
        del timestep, state
        raise NotImplementedError()

    def initial_state(self, batch_size: Optional[int]):
        """Generates an initial state for the policy."""
        return PREPState(
            recurrent_state=self._memory_core.initial_state(batch_size),
            logits=np.zeros([self._policy_head.num_actions], dtype=np.float32),
            # No previous action is denoted as -1, which will give an all zero one-hot.
            prev_action=np.array(-1, dtype=np.int32),
        )

    def state_spec(self) -> worlds.TreeSpec:
        """Specification describing the state of the policy."""
        return spec_utils.make_tree_spec(self.initial_state(None))

    def unroll(self, timestep: worlds.TimeStep, state_and_extras: PREPState):
        """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
        del timestep, state_and_extras
        raise NotImplementedError()

    def loss(self, data: reverb_adder.Step) -> Tuple[_types.Tree, loggers.LogData]:
        """."""
        del data
        raise NotImplementedError()

    def bc_loss(self, data: reverb_adder.Step) -> Tuple[_types.Tree, loggers.LogData]:
        """Behavioural cloning loss.

        Args:
            data: Batch of training data. BC expects the observation to include:
                * demonstration: The action taking by the demonstrating agent.
                * self_id: The ID of the demonstrator.
                * padding_mask: Boolean mask indicating if the example is real (T) or padding (F).

        """
        padding_mask = data.observation["padding_mask"]  # [B, T].
        metrics = {}

        # The target of our BC loss is the action taken by the demonstrator.
        target = data.observation["demonstration"]  # [B, T] with the demonstrator's action.
        target = jax.nn.one_hot(target, self._num_actions)  # [B, T, A].

        # Tell the policy demonstrator to clone.
        self_id = data.observation["self_id"]
        self_id = hk.BatchApply(self._id_embedder)(self_id)
        latent = self_id  # [B, T, #Demonstrators].

        init_state = self._conditional_policy.initial_state(target.shape[0])
        pred, _ = self._conditional_policy.unroll(data, latent, init_state)

        loss = optax.softmax_cross_entropy(logits=pred, labels=target)  # [B, T].
        loss = jnp.mean(loss * padding_mask)  # NOTE: This averaging does not consider padded arrays.
        metrics["loss"] = loss

        metrics.update(self._bc_accuracy_metrics(pred, data.observation["demonstration"], padding_mask))
        return loss, metrics

    def _bc_accuracy_metrics(
        self, logits: _types.Array, targets: _types.Array, padding_mask: _types.Array
    ) -> loggers.LogData:
        """Compute accuracy metrics for BC.

        Args:
            predictions: Predicted demonstrator's logits [B, T, A].
            targets: Demonstrator's taken action [B, T].
            padding_mask: Mask flagging if an example is padding [B, T].
        """
        metrics = {}

        # Greedy metrics: assume that the action taken was deterministically sampled as the argmax.
        greedy = jnp.argmax(logits, axis=-1)
        greedy_correct = greedy == targets
        metrics["accuracy/greedy"] = jnp.mean(greedy_correct * padding_mask)

        # Sampled metrics: assume that the action taken was sampled.
        sampled = jax.random.categorical(hk.next_rng_key(), logits)  # [B, T, A] --> [B, T].
        sampled_correct = sampled == targets
        metrics["accuracy/sampled"] = jnp.mean(sampled_correct * padding_mask)

        # Per-action metrics.
        for i in range(self._num_actions):
            this_action = targets == i
            num_class = jnp.sum(this_action * padding_mask)

            # Greedy.
            greedy_correct_and_this_action = jnp.logical_and(greedy_correct, this_action)
            metrics[f"accuracy/greedy/{i}"] = jnp.sum(greedy_correct_and_this_action * padding_mask) / num_class

            # Sampled.
            sampled_correct_and_this_action = jnp.logical_and(sampled_correct, this_action)
            metrics[f"accuracy/sampled/{i}"] = jnp.sum(sampled_correct_and_this_action * padding_mask) / num_class

        return metrics


class OneHot(hk.Module):
    """One-hot embedder."""

    def __init__(self, num_classes: int, name: Optional[str] = "one_hot"):
        """Initializes an instance of an OneHot module."""
        super().__init__(name=name)
        self._num_classes = num_classes

    def __call__(self, x: _types.Array) -> _types.Array:
        """Forward pass of the module."""
        return jax.nn.one_hot(x, num_classes=self._num_classes)


class Concatenate(hk.Module):
    """Concatenates the latent and observation embedding.."""

    def __init__(self, name: Optional[str] = "concatenate"):
        """Initializes an instance of a Concatenate module."""
        super().__init__(name=name)

    def __call__(self, observation: _types.Array, latent: _types.Array) -> _types.Tree:
        """Forward pass of the module."""
        return jnp.concatenate([observation, latent], axis=-1)


class ConditionedPolicy(hk.Module):
    def __init__(
        self,
        timestep_encoder: hk.Module,
        fusion_method: hk.Module,
        memory_core: hk.Module,
        policy_head: hk.Module,
        name: Optional[str] = "conditioned_policy",
    ):
        super().__init__(name=name)
        self._timestep_encoder = timestep_encoder
        self._fusion_method = fusion_method
        self._memory_core = memory_core
        self._policy_head = policy_head

    def __call__(
        self, timestep: worlds.TimeStep, latent: _types.Array, state: _types.State
    ) -> Tuple[_types.Tree, _types.State]:
        embeddings = self._timestep_encoder(timestep, state)
        embeddings = self._fusion_method(embeddings, latent)
        embeddings, new_recurrent_state = self._memory_core(embeddings, state.recurrent_state)
        logits = self._policy_head(embeddings)
        return logits, new_recurrent_state

    def initial_state(self, batch_size: Optional[int]) -> _types.State:
        return self._memory_core.initial_state(batch_size)

    def unroll(
        self, timestep: worlds.TimeStep, latents: _types.Array, state_and_extras: _types.State
    ) -> Tuple[_types.Tree, _types.State]:
        """This should be for additional time dimension over call"""
        embeddings = hk.BatchApply(self._timestep_encoder)(timestep, state_and_extras)
        embeddings = hk.BatchApply(self._fusion_method)(embeddings, latents)
        embeddings, new_states = self._memory_core.unroll(embeddings, state_and_extras)
        logits = hk.BatchApply(self._policy_head)(embeddings)
        return logits, new_states
