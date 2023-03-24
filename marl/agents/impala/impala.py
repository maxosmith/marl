import functools
from typing import Mapping, NamedTuple, Optional, Tuple

import chex
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
from marl.utils import loggers, spec_utils, stats_utils


class IMPALAState(NamedTuple):
    """Recurrent state and extras of the IMPALA agent."""

    recurrent_state: _types.Tree
    logits: _types.Logits
    value: _types.Array
    prev_action: _types.Action


class IMPALA(hk.RNNCore):
    """IMPALA agent graphs."""

    def __init__(
        self,
        *,
        # Sub-modules.
        timestep_encoder: hk.Module,
        memory_core: hk.Module,
        policy_head: hk.Module,
        value_head: hk.Module,
        evaluation: bool,
        # Hyperparameters.
        discount: float = 0.99,
        max_abs_reward: float = np.inf,
        baseline_cost: float = 0.5,
        entropy_cost: float = 0.02,
        policy_cost: float = 1.0,
        name: Optional[str] = "impala",
    ):
        """Initialize an instance of the IMPALA algorithm's computational graphs.

        Args:
            timestep_encoder: Module that learns an embedding for a timestep.
            memory_core: Module responsible for maintaining state/memory.
            policy_head: Module that predicts per-action logits.
            value_head: Module that predicts state values.
            evaluation: Whether action selection should occur with exploration (True) or without (False).
            baseline_cost: Baseline cost coefficient ().
            entropy_cost: Entropy cost coefficient ().
            policy_cost: Policy cost coefficient ().
        """
        super().__init__(name=name)
        self._discount = discount
        self._max_abs_reward = max_abs_reward
        self._baseline_cost = baseline_cost
        self._entropy_cost = entropy_cost
        self._policy_cost = policy_cost

        self._timestep_encoder = timestep_encoder
        self._memory_core = memory_core
        self._policy_head = policy_head
        self._value_head = value_head
        self._evaluation = evaluation

        if not hasattr(self._policy_head, "num_actions"):
            raise ValueError("Policy head must have attribute `num_actions`.")

    def __call__(self, timestep: worlds.TimeStep, state: IMPALAState) -> Tuple[_types.Action, IMPALAState]:
        """Forward pass of IMPALA's policy."""
        embeddings = self._timestep_encoder(timestep, state)
        embeddings, new_recurrent_state = self._memory_core(embeddings, state.recurrent_state)
        logits = self._policy_head(embeddings)
        value = self._value_head(embeddings)
        if self._evaluation:
            action = jnp.argmax(logits, axis=-1)
        else:
            action = hk.multinomial(hk.next_rng_key(), logits, num_samples=1)[0]
        return action, IMPALAState(recurrent_state=new_recurrent_state, logits=logits, value=value, prev_action=action)

    def initial_state(self, batch_size: Optional[int]):
        """Generates an initial state for the policy."""
        return IMPALAState(
            recurrent_state=self._memory_core.initial_state(batch_size),
            logits=np.zeros([self._policy_head.num_actions], dtype=np.float32),
            value=np.array(0.0, dtype=np.float32),
            # No previous action is denoted as -1, which will give an all zero one-hot.
            prev_action=np.array(-1, dtype=np.int32),
        )

    def state_spec(self) -> worlds.TreeSpec:
        """Specification describing the state of the policy."""
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

        We compute the loss over batches of sequences (i.e., shape [B, T, ...]). This involes some
        hairy notation relating to the time axis that we briefly explain here. Below is the high-level
        structure of a single example:

                    T   0 1 2 3 4 5 6 7
                        o o o o o o o o
                        a a a a a a a a
                        r r r r r r r r

        This sequence can be from any sub-trajectory within an episode and may be left zero-padded.
        Since the episode may continue past sub-trajectory we can hold in RAM, we must compute a bootstrap
        value as an estimate of the future return. For each observation we compute the target-logits and
        baseline value of the future. We also now add in two notions of time: past and current. This is
        because the last step in the sequence / end of episodes are padded since we don't know the future:

                        |-----------|           tm1: time minus 1
                          |-----------|         t: time
                    T   0 1 2 3 4 5 6 7
                        l l l l l l l l         l: logits computed for all observations.
                        v v v v v v v v         v: baseline value estimate.
                        a a a a a a a a
                        r r r r r r r r

        Rewritten:

                    T   0 1 2 3 4 5 6 7
                        l l l l l l l l
                        v v v v v v v           tm1
                          v v v v v v v         t
                        a a a a a a a a
                        r r r r r r r r

        Data is stored so that (o_t, a_t, r_{t+1}) are actually lined up in a column. This means that
        the last timestep in an episode writes the (a, r) for the previous column, and has the final
        observation that is put in a padded column. We can use this observation as our bootstrap value,
        but need to remove the dummy values. Moreover, we trim off the last logits because we do not
        have the associated labels but instead the dummy values.

                    T   0 1 2 3 4 5 6 7
                        l l l l l l l
                        v v v v v v v           tm1
                          v v v v v v v         t
                        a a a a a a a
                        r r r r r r r

        In summary, the last entry in the sequence is only used as an observation for a bootstrapped value.

        Args:
            data: A batch of Sequence data, following the `Step` structure with leading
                dimensions: [B, T, ...].

        Returns:
            IMPALA loss and logging metrics.
        """
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
        padding_mask = reverb_utils.padding_mask(data)

        # Apply reward clipping.
        rewards = jnp.clip(rewards, -self._max_abs_reward, self._max_abs_reward)

        # Unroll current policy over observations.
        (logits, values), _ = self.unroll(data, state_and_extras)

        # Apply loss function over T.
        loss_fn = functools.partial(
            impala_loss,
            discount=self._discount,
            baseline_cost=self._baseline_cost,
            entropy_cost=self._entropy_cost,
            policy_cost=self._policy_cost,
        )

        # Seperate the bootstrap from the value estimates.
        baseline_tm1 = values[:, :-1]
        baseline_t = values[:, 1:]

        # Remove bootstrap timestep from non-observations.
        actions = actions[:, :-1]
        logits = logits[:, :-1]
        behaviour_logits = behaviour_logits[:, :-1]
        padding_mask = padding_mask[:, :-1]
        rewards = rewards[:, :-1]

        mean_loss, metrics = jax.vmap(loss_fn, in_axes=0)(  # Apply loss batch-wise.
            logits=logits,
            behaviour_logits=behaviour_logits,
            actions=actions,
            values_tm1=baseline_tm1,
            values_t=baseline_t,
            rewards=rewards,
            mask=padding_mask,
        )
        metrics = tree.map_structure(jnp.mean, metrics)

        metrics["batch_size"] = actions.shape[0]

        # Log the average policy to monitor distribution collapse.
        policy = jax.nn.softmax(logits)
        metrics["policy_most_likely_action"] = jnp.mean(jnp.max(policy, axis=-1))
        metrics["policy"] = policy

        # Log the average values across the sequence.
        metrics["value"] = jnp.mean(values, axis=0)  # [B, T] --> [T].

        mean_loss = jnp.mean(mean_loss)
        return mean_loss, metrics


def impala_loss(
    logits: _types.Array,
    behaviour_logits: _types.Array,
    actions: _types.Array,
    values_tm1: _types.Array,
    values_t: _types.Array,
    rewards: _types.Array,
    mask: _types.Array,
    *,
    discount,
    baseline_cost,
    entropy_cost,
    policy_cost,
) -> Tuple[_types.Array, loggers.LogData]:
    """IMPALA loss applied to sequential transitions.

    Args:
        logits: Logits for the target policy [T, A]
        behaviour_logits: Logits for the behaviour policy [T, A].
        values_tm1: State values for t-1 [T].
        values_t: State values for t [T].
        rewards: Rewards [T].
        mask: Mask [T].
        discount: Discount factor ().
        baseline_cost: Baseline cost coefficient ().
        entropy_cost: Entropy cost coefficient ().
        policy_cost: Policy cost coefficient ().

    Returns:
        Mean loss and additional metrics for logging.
    """
    chex.assert_rank([logits, behaviour_logits, actions, values_tm1, values_t, rewards, mask], [2, 2, 1, 1, 1, 1, 1])
    chex.assert_type([logits, behaviour_logits, values_tm1, values_t, rewards], float)
    chex.assert_type([actions, mask], [int, bool])

    # Compute importance sampling weights: current policy / behavior policy.
    rhos = rlax.categorical_importance_sampling_ratios(logits, behaviour_logits, actions)

    # Critic loss.
    vtrace_returns = rlax.vtrace_td_error_and_advantage(
        v_tm1=values_tm1,
        v_t=values_t,
        r_t=rewards,
        discount_t=jnp.full_like(rewards, discount),
        rho_tm1=rhos,
    )
    critic_loss = jnp.square(vtrace_returns.errors)
    critic_loss = critic_loss * mask
    critic_loss = jnp.mean(critic_loss)

    # Policy gradient loss.
    policy_gradient_loss = rlax.policy_gradient_loss(
        logits_t=logits,
        a_t=actions,
        adv_t=vtrace_returns.pg_advantage,
        w_t=mask.astype(float),
    )
    policy_gradient_loss = jnp.mean(policy_gradient_loss)

    # Entropy regulariser.
    entropy_loss = rlax.entropy_loss(logits, w_t=mask.astype(float))

    # Combine weighted sum of actor & critic losses, averaged over the sequence.
    mean_loss = policy_cost * policy_gradient_loss + baseline_cost * critic_loss + entropy_cost * entropy_loss
    metrics = {
        "loss": mean_loss,
        "policy_loss": policy_gradient_loss,
        "scaled_policy_loss": policy_cost * policy_gradient_loss,
        "critic_loss": critic_loss,
        "scaled_critic_loss": baseline_cost * critic_loss,
        "entropy_loss": entropy_loss,
        "scaled_entropy_loss": entropy_cost * entropy_loss,
        "critic_explained_variance": stats_utils.explained_variance(
            y=rewards[..., None] + discount * values_t[..., None],  # TD target.
            pred=values_tm1[..., None],
        ),
        "kl_vs_target": optax._src.loss.kl_divergence(
            log_predictions=jnp.log(jax.nn.softmax(logits)),
            targets=jax.nn.softmax(behaviour_logits),
        ),
    }
    return mean_loss, metrics
