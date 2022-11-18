"""Train an agent through Behavioural Cloning using demonstration data from bots."""
from typing import Any, Optional, Sequence

import haiku as hk
import numpy as np
import optax
import tree
from absl import app
from ml_collections import config_dict
from torch.utils.data import DataLoader

from marl import services, utils, worlds
from marl.utils import loggers as logger_lib
from marl_experiments.roshambo import networks, roshambo_bot
from marl_experiments.roshambo.agents import prep
from marl_experiments.roshambo.services import behavioural_cloning as bc_services
from marl_experiments.roshambo.utils import bc_dataset


def get_config() -> config_dict.ConfigDict:
    """Default configuraiton for the experiment."""
    return config_dict.create(
        result_dir="/scratch/wellman_root/wellman1/mxsmith/results/roshambo/test_bc/",
        seed=42,
        num_learner_steps=1_000_000,
        # Constructor arguments for a `BCDataset`.
        dataset_config=config_dict.create(
            path="/scratch/wellman_root/wellman1/mxsmith/data/roshambo/demonstrations2.zarr",
            bot_names=["rotatebot"],
            # Opponent names are assumed the same as the bots when set as None.
            opponent_names=roshambo_bot.ROSHAMBO_BOT_NAMES,
            sequence_len=20,
        ),
        # Constructor arguments for a `DataLoader`.
        dataloader_config=config_dict.create(
            batch_size=64,
            shuffle=True,
            num_workers=4,
        ),
        optimizer=config_dict.create(
            learning_rate=0.003,
            max_gradient_norm=10,
        ),
    )


def build_dataloader(dataset_config, dataloader_config):
    """Builds a dataloader for the BC datasett."""

    def collate_fn(data: Sequence[Any]) -> Any:
        """Stacks a batch of data."""
        return tree.map_structure(lambda *x: np.stack(x), *data)

    # If opponents aren't listed seperately, assume they're the same set as the bots.
    if ("opponent_names" not in dataset_config) or dataset_config["opponent_names"] is None:
        dataset_config["opponent_names"] = dataset_config["bot_names"]

    dataset = bc_dataset.BCDataset(**dataset_config)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, **dataloader_config)
    return dataloader


def build_graphs():
    """Build agent graphs."""

    def _build(x):
        """Build utility function for closure."""

        policy = prep.ConditionedPolicy(
            timestep_encoder=networks.MLPTimestepEncoder(),
            fusion_method=prep.Concatenate(),
            memory_core=networks.MemoryCore(),
            policy_head=networks.PolicyHead(num_actions=3),
        )

        agent = prep.PREP(
            conditional_policy=policy,
            evidence_encoder=None,
            id_embedder=prep.OneHot(num_classes=1),
            best_responder=None,
            num_actions=3,
        )

        return agent.bc_loss(x)

    return hk.transform(_build)


def run(config: Optional[config_dict.ConfigDict] = None):
    """Run BC."""
    if config is None:
        config = get_config()
    key_sequence = hk.PRNGSequence(config.seed)
    result_dir = utils.ResultDirectory(config.result_dir, overwrite=True, exist_ok=True)

    dataloader = build_dataloader(config.dataset_config, config.dataloader_config)

    bc_loss = build_graphs()

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.optimizer.max_gradient_norm),
        optax.adam(learning_rate=config.optimizer.learning_rate),
    )

    logger = logger_lib.LoggerManager(
        loggers=[
            logger_lib.TerminalLogger(time_frequency=5),
            logger_lib.TensorboardLogger(result_dir.dir),
        ],
        time_frequency=5,  # Seconds.
    )
    learner = bc_services.Learner(
        loss_fn=bc_loss,
        optimizer=optimizer,
        key_sequence=key_sequence,
        data_iterator=iter(dataloader),
        logger=logger,
        counter=services.Counter(),
    )

    for _ in range(config.num_learner_steps):
        learner.step()


def main(_):
    """Enables running the file directly through absl, and also running with a config input."""
    run()


if __name__ == "__main__":
    app.run(main)
