from typing import NamedTuple

import haiku as hk
import jax
import numpy as np
import optax
import reverb
import tree
from absl import app
from marl_experiments.gathering import networks

from marl import games, services
from marl.agents.impala import impala
from marl.services.replay.reverb import adders
from marl.services.replay.reverb.adders import reverb_adder
from marl.utils import wrappers


def main(_):
    game = games.Gathering(n_agents=1, global_observation=True, map_name="default_small")
    game = wrappers.TimeLimit(game, num_steps=11)

    trajectory = []
    timestep_trajectory = []
    timesteps = game.reset()
    timestep_trajectory.append(timesteps[0])

    # ==============================================================================================
    def _build_impala():
        impala = impala.IMPALA(
            timestep_encoder=networks.MLPTimestepEncoder(num_actions=8),
            memory_core=networks.MemoryLessCore(),
            policy_head=networks.PolicyHead(num_actions=8),
            value_head=networks.ValueHead(),
            evaluation=False,
        )

        def init():
            return impala(timesteps[0], impala.initial_state(None))

        return init, (impala.__call__, impala.loss, impala.initial_state, impala.state_spec)

    impala_graphs = hk.multi_transform(_build_impala)
    state_spec = impala_graphs.apply[3](None, None)

    random_key = jax.random.PRNGKey(42)

    random_key, subkey = jax.random.split(random_key)
    params = impala_graphs.init(subkey)

    random_key, subkey = jax.random.split(random_key)
    state = impala_graphs.apply[2](None, subkey, None)

    # Same random key should always sample the same action.
    random_key, subkey = jax.random.split(random_key)
    print(impala_graphs.apply[0](params, subkey, timesteps[0], state))
    print(impala_graphs.apply[0](params, subkey, timesteps[0], state))

    # New random key, odds are a different action will be sampled.
    random_key, subkey = jax.random.split(random_key)
    print(impala_graphs.apply[0](params, subkey, timesteps[0], state))

    # ==============================================================================================

    # Initialize the reverb server.
    signature = adders.SequenceAdder.signature(
        environment_spec=game.spec()[0],
        extras_spec=state_spec,
        sequence_length=11,
    )
    reverb_server = reverb.Server(
        tables=[
            reverb.Table(
                name="trajectories",
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=100,
                rate_limiter=reverb.rate_limiters.MinSize(1),
                signature=signature,
            )
        ],
    )
    reverb_client = reverb.Client(f"localhost:{reverb_server.port}")
    adder = adders.SequenceAdder(
        client=reverb_client,
        priority_fns={"trajectories": None},
        sequence_length=11,
        period=11,
    )

    # ==============================================================================================

    print(game.render(mode="text"))
    adder.add(timestep=timesteps[0])
    history = get_adder_history(adder)

    for _ in range(4):
        new_timesteps = game.step({0: games.GatheringActions.DOWN})
        random_key, subkey = jax.random.split(random_key)
        _, extras = impala_graphs.apply[0](params, subkey, timesteps[0], state)
        trajectory.append(
            reverb_adder.Step(
                observation=timesteps[0].observation,
                action=games.GatheringActions.DOWN,
                reward=new_timesteps[0].reward,
                start_of_episode=timesteps[0].first(),
                end_of_episode=timesteps[0].last(),
                extras=extras,
            )
        )
        timesteps = new_timesteps
        timestep_trajectory.append(timesteps[0])
        adder.add(action=np.asarray(games.GatheringActions.DOWN, np.int32), timestep=timesteps[0], extras=extras)
        print(game.render(mode="text"))
        history = get_adder_history(adder)

    for i in range(7):
        new_timesteps = game.step({0: games.GatheringActions.RIGHT})
        random_key, subkey = jax.random.split(random_key)
        _, extras = impala_graphs.apply[0](params, subkey, timesteps[0], state)
        trajectory.append(
            reverb_adder.Step(
                observation=timesteps[0].observation,
                action=games.GatheringActions.RIGHT,
                reward=new_timesteps[0].reward,
                start_of_episode=timesteps[0].first(),
                end_of_episode=timesteps[0].last(),
                extras=extras,
            )
        )
        timesteps = new_timesteps
        timestep_trajectory.append(timesteps[0])
        adder.add(action=np.asarray(games.GatheringActions.RIGHT, np.int32), timestep=timesteps[0], extras=extras)
        print(i)
        print(game.render(mode="text"))
        history = get_adder_history(adder)

    # ==============================================================================================

    trajectory: reverb_adder.Step = tree.map_structure(lambda *x: np.stack(x), *trajectory)
    print(trajectory.reward)
    print(trajectory.start_of_episode)
    print(trajectory.end_of_episode)

    # Add a batch dimension.
    trajectory: reverb_adder.Step = tree.map_structure(lambda x: np.expand_dims(x, axis=0), trajectory)

    # Optimizer.
    optimizer = optax.sgd(1e-4)

    def _update_step(params, rng, opt_state, data):
        grad_fn = jax.value_and_grad(impala_graphs.apply[1], has_aux=True)
        (_, metrics), gradients = grad_fn(params, rng, data)
        updates, new_opt_state = optimizer.update(gradients, opt_state)
        new_params = optax.apply_updates(params, updates)
        return metrics, new_params, new_opt_state

    _update_step = jax.jit(_update_step)
    opt_state = optimizer.init(params)

    # # Loss computation.
    # for _ in range(1_000):
    #     random_key, subkey = jax.random.split(random_key)
    #     metrics, params, opt_state = _update_step(params, subkey, opt_state, trajectory)
    #     print(metrics["loss"])
    #     # print(metrics)
    # print(metrics)

    timesteps = game.reset()
    random_key, subkey = jax.random.split(random_key)
    action, extras = impala_graphs.apply[0](params, subkey, timesteps[0], state)
    print(game.render(mode="text"))
    print(action)
    print(extras)

    # ==============================================================================================

    data_iterator = services.ReverbPrefetchClient(
        reverb_client=reverb_client,
        table_name="trajectories",
        batch_size=1,
    )

    for _ in range(10):
        batch = next(data_iterator)

        print("")
        print("Start:  ", batch.data.start_of_episode.astype(int))
        print("End:    ", batch.data.end_of_episode.astype(int))
        print("Reward: ", batch.data.reward.astype(int))
        print("# Samp: ", batch.info.times_sampled)

    # Remove the device dimension since we're not distributing the update.
    batch = tree.map_structure(lambda x: np.squeeze(x, axis=0), batch.data)

    # Loss computation.
    for _ in range(1_000):
        random_key, subkey = jax.random.split(random_key)
        metrics, params, opt_state = _update_step(params, subkey, opt_state, batch)
        print(metrics["loss"])
        # print(metrics)
    print(metrics)

    timesteps = game.reset()
    random_key, subkey = jax.random.split(random_key)
    action, extras = impala_graphs.apply[0](params, subkey, timesteps[0], state)
    print(game.render(mode="text"))
    print(action)
    print(extras)


def get_adder_history(adder):
    base_history = adder._writer.history
    # Get the internal references to the data in C.
    history = tree.map_structure(lambda x: x._data_references, base_history)
    # Convert the data into Python numpy objects, ignoring entries in partial rows.
    history = tree.map_structure(lambda x: x.numpy() if x else x, history)
    # Numpy-ify the lists of numpy objects (which was original just lists of references).
    history = tree.map_structure_up_to(base_history, lambda *x: np.stack(x), history)
    return history


def _validate_gathering_data(data):
    # data.
    #     .action                       (1, 32, 20).
    #     .start_of_episode             (1, 32, 20).
    #     .end_of_episode               (1, 32, 20).
    #     .observation                  (1, 32, 20, 21, 19, 5).
    #     .reward                       (1, 32, 20).
    #     .extras.
    #            .recurrent_state
    #            .logits                (1, 32, 20, 8).
    #            .prev_action           (1, 32, 20).

    D, B, T = data.action.shape

    padding_mask = reverb_utils.padding_mask(data)

    # Verify that if an `end_of_episode` is triggered, that a `start_of_episode`
    # does not occur after it. Otherwise, we need to include this in our masking process.
    for d, b, t in itertools.product(range(D), range(B), range(T)):
        if t == 0:
            print("Start:  ", data.start_of_episode[d, b].astype(int))
            print("End:    ", data.end_of_episode[d, b].astype(int))
            print("Mask:   ", padding_mask[d, b].astype(int))
            print("Action: ", data.action[d, b].astype(int))
            print("Reward: ", data.reward[d, b].astype(int))
            print(" ")

        if data.end_of_episode[d, b, t]:
            assert np.sum(data.start_of_episode[d, b, t:]) == 0

    print("Reward histogram:")
    print(np.sum(data.reward, axis=[0, 1]))


if __name__ == "__main__":
    app.run(main)
