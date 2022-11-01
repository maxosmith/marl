from typing import Callable, Mapping, NamedTuple, Optional, Tuple, Union

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import rlax
import tree
from jax import lax

from marl import _types, worlds
from marl.services.replay.reverb.adders import reverb_adder
from marl.services.replay.reverb.adders import utils as reverb_utils
from marl.utils import loggers, spec_utils


class DQNState(NamedTuple):
    recurrent_state: _types.Tree
    prev_action: _types.Action
    num_steps: int


class DQN(hk.RNNCore):
    def __init__(
        self,
        *,
        # Sub-modules.
        timestep_encoder: hk.Module,
        memory_core: hk.Module,
        value_head: hk.Module,
        evaluation: bool,
        # Hyperparameters.
        epsilon: Union[float, Callable[[int], float]] = 0.05,
        discount: float = 0.99,
        max_abs_reward: float = 1.0,
        huber_loss_parameter: float = 1.0,
        name: Optional[str] = "dqn",
    ):
        """Initializes an instance of the DQN algorithms computaitonal graphs.

        Args:
            timestep_encoder: Module that learns an embedding for a timestep.
            memory_core: Module responsible for maintaining state/memory.
            value_head: Module that predicts action values.
            evaluation: Whether action selection should occur with exploration (True) or without (False).
        """
        super().__init__(name=name)
        self._timestep_encoder = timestep_encoder
        self._memory_core = memory_core
        self._value_head = value_head
        self._evaluation = evaluation

        self._epsilon = epsilon
        self._discount = discount
        self._max_abs_reward = max_abs_reward
        self._huber_loss_parameter = huber_loss_parameter

        if not hasattr(self._value_head, "num_actions"):
            raise ValueError("Value head must have attribute `num_actions`.")

    def __call__(self, timestep: worlds.TimeStep, state: DQNState) -> Tuple[_types.Action, DQNState]:
        embeddings = self._timestep_encoder(timestep, state)
        embeddings, new_recurrent_state = self._memory_core(embeddings, state.recurrent_state)
        q_values = self._value_head(embeddings)
        if self._evaluation:
            action = jnp.argmax(q_values, axis=-1)
        else:
            epsilon = self._epsilon if isinstance(self._epsilon, float) else self._epsilon(state.num_steps)
            action = distrax.EpsilonGreedy(q_values, epsilon).sample(seed=hk.next_rng_key())
        new_state = DQNState(
            recurrent_state=new_recurrent_state,
            prev_action=action,
            num_steps=state.num_steps + 1,
        )
        return action, new_state

    def initial_state(self, batch_size: Optional[int]):
        return DQNState(
            recurrent_state=self._memory_core.initial_state(batch_size),
            # No previous action is denoted as -1, which will give an all zero one-hot.
            prev_action=np.array(-1, dtype=np.int32),
            num_steps=np.array(0, dtype=np.int32),
        )

    def state_spec(self) -> worlds.TreeSpec:
        return spec_utils.make_tree_spec(self.initial_state(None))

    def unroll(self, timestep: worlds.TimeStep, state_and_extras: DQNState):
        """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
        embeddings = hk.BatchApply(self._timestep_encoder)(timestep, state_and_extras)
        embeddings, new_states = self._memory_core.unroll(embeddings, state_and_extras.recurrent_state)
        values = hk.BatchApply(self._value_head)(embeddings)
        return values, new_states

    def loss(self, data: reverb_adder.Step) -> Tuple[_types.Tree, Mapping[str, _types.Tree]]:
        # Extract the data.
        _, actions, rewards, state_and_extras = (
            data.observation,
            data.action,
            data.reward,
            data.extras,
        )
        initial_state = tree.map_structure(lambda s: s[:, 0], state_and_extras.recurrent_state)
        state_and_extras = state_and_extras._replace(recurrent_state=initial_state)
        padding_mask = reverb_utils.padding_mask(data)

        # Apply reward clipping.
        rewards = jnp.clip(rewards, -self._max_abs_reward, self._max_abs_reward)

        # Unroll current policy over observations.
        values, _ = self.unroll(data, state_and_extras)

        # DQN loss, applied over both B and T.
        q_learning = jax.vmap(rlax.q_learning)
        td_error = hk.BatchApply(q_learning)(
            q_tm1=values[:, :-1],
            a_tm1=actions[:, :-1],
            r_t=rewards[:, :-1],
            q_t=values[:, 1:],  # rlax applies stop gradient.
            discount_t=jnp.full_like(rewards[:, :-1], self._discount),
        )
        td_error = td_error * padding_mask[:, :-1]
        loss = rlax.huber_loss(td_error, delta=self._huber_loss_parameter)
        loss = jnp.mean(loss)

        metrics = dict(
            loss=loss,
            batch_size=actions.shape[0],
        )
        return loss, metrics
