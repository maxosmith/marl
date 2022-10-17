from typing import Callable, Mapping, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import reverb
import rlax
import tree

from marl import _types, worlds
from marl.rl.replay.reverb.adders import reverb_adder
from marl.utils import spec_utils


class IMPALAState(NamedTuple):
    recurrent_state: _types.Tree
    logits: _types.Logits
    prev_action: _types.Action


class IMPALA(hk.RNNCore):
    def __init__(
        self,
        *,
        # Sub-modules.
        timestep_encoder: hk.Module,
        memory_core: hk.Module,
        policy_head: hk.Module,
        value_head: hk.Module,
        # Hyperparameters.
        discount: float = 0.99,
        max_abs_reward: float = np.inf,
        baseline_cost: float = 1.0,
        entropy_cost: float = 0.0,
        name: Optional[str] = "impala",
    ) -> None:
        """Initialize an instance of the IMPALA algorithm's computational graphs.

        Args:
            discount: The standard geometric discount rate to apply.
            max_abs_reward: Optional symmetric reward clipping to apply.
            baseline_cost: Weighting of the critic loss relative to the policy loss.
            entropy_cost: Weighting of the entropy regulariser relative to policy loss.
        """
        super().__init__(name=name)
        self._discount = discount
        self._max_abs_reward = max_abs_reward
        self._baseline_cost = baseline_cost
        self._entropy_cost = entropy_cost

        self._timestep_encoder = timestep_encoder
        self._memory_core = memory_core
        self._policy_head = policy_head
        self._value_head = value_head

        if not hasattr(self._policy_head, "num_actions"):
            raise ValueError("Policy head must have attribute `num_actions`.")

    def __call__(self, timestep: worlds.TimeStep, state: IMPALAState) -> _types.Action:
        embeddings = self._timestep_encoder(timestep, state)
        embeddings, new_recurrent_state = self._memory_core(embeddings, state.recurrent_state)
        logits = self._policy_head(embeddings)
        _ = self._value_head(embeddings)  # Calculate values to build associated parameters.
        action = jnp.argmax(logits, axis=-1)
        return action, IMPALAState(recurrent_state=new_recurrent_state, logits=logits, prev_action=action)

    def initial_state(self, batch_size: Optional[int]):
        return IMPALAState(
            recurrent_state=self._memory_core.initial_state(batch_size),
            logits=np.zeros([self._policy_head.num_actions], dtype=np.float32),
            # No previous action is denoted as -1, which will give an all zero one-hot.
            prev_action=np.array(-1, dtype=np.int32),
        )

    def state_spec(self) -> worlds.TreeSpec:
        return spec_utils.make_tree_spec(self.initial_state(None))

    def unroll(self, timestep: worlds.TimeStep, state_and_extras: IMPALAState):
        """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
        embeddings = hk.BatchApply(self._timestep_encoder)(timestep, state_and_extras)
        embeddings, new_states = self._memory_core.unroll(embeddings, state_and_extras.recurrent_state)
        logits = hk.BatchApply(self._policy_head)(embeddings)
        values = hk.BatchApply(self._value_head)(embeddings)
        return (logits, values), new_states

    def loss(self, data: reverb_adder.Step) -> Tuple[_types.Tree, Mapping[str, _types.Tree]]:
        """Builds the standard entropy-regularised IMPALA loss function.

        TODO(maxsmith): Replace Step with a more standard TimeStep/Trajectory.

        Args:
            data: A batch of Sequence data, following the `Step` structure with leading
                dimensions: [B, T, ...].

        Returns:
            IMPALA loss and logging metrics.
        """

        def _impala_loss(logits, behaviour_logits, actions, values_tm1, values_t, rewards):
            """IMPALA loss applied to non-sequential transition."""
            # Compute importance sampling weights: current policy / behavior policy.
            rhos = rlax.categorical_importance_sampling_ratios(logits, behaviour_logits, actions)

            # Critic loss.
            vtrace_returns = rlax.vtrace_td_error_and_advantage(
                v_tm1=values_tm1,
                v_t=values_t,
                r_t=rewards,
                discount_t=jnp.full_like(rewards, self._discount),
                rho_tm1=rhos,
            )
            critic_loss = jnp.square(vtrace_returns.errors)

            # Policy gradient loss.
            policy_gradient_loss = rlax.policy_gradient_loss(
                logits_t=logits,
                a_t=actions,
                adv_t=vtrace_returns.pg_advantage,
                w_t=jnp.ones_like(rewards),
            )

            # Entropy regulariser.
            entropy_loss = rlax.entropy_loss(logits, jnp.ones_like(rewards))

            # Combine weighted sum of actor & critic losses, averaged over the sequence.
            mean_loss = jnp.mean(
                policy_gradient_loss + self._baseline_cost * critic_loss + self._entropy_cost * entropy_loss
            )

            metrics = {
                "loss": mean_loss,
                "policy_loss": jnp.mean(policy_gradient_loss),
                "critic_loss": jnp.mean(self._baseline_cost * critic_loss),
                "scaled_critic_loss": jnp.mean(critic_loss),
                "entropy_loss": jnp.mean(entropy_loss),
                "scaled_entropy_loss": jnp.mean(self._entropy_cost * entropy_loss),
                "entropy": jnp.mean(entropy_loss),
            }
            return mean_loss, metrics

        # Extract the data.
        _, actions, rewards, state_and_extras = (
            data.observation,
            data.action,
            data.reward,
            data.extras,
        )
        initial_state = tree.map_structure(lambda s: s[:, 0], state_and_extras.recurrent_state)
        state_and_extras = state_and_extras._replace(recurrent_state=initial_state)
        behaviour_logits = state_and_extras.logits

        # Apply reward clipping.
        rewards = jnp.clip(rewards, -self._max_abs_reward, self._max_abs_reward)

        # Unroll current policy over observations.
        (logits, values), _ = self.unroll(data, state_and_extras)

        # Apply loss function over T.
        loss_fn = jax.vmap(_impala_loss, in_axes=1)
        mean_loss, metrics = loss_fn(
            logits=logits[:, :-1],
            behaviour_logits=behaviour_logits[:, :-1],
            actions=actions[:, :-1],
            values_tm1=values[:, :-1],
            values_t=values[:, 1:],
            rewards=rewards[:, :-1],
        )
        mean_loss = jnp.mean(mean_loss)
        metrics = tree.map_structure(jnp.mean, metrics)
        metrics["batch_size"] = actions.shape[0]
        return mean_loss, metrics
