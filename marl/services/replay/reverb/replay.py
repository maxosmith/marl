"""Tools for reverb"""
import dataclasses
from typing import Callable, Iterator, Optional

import reverb

from marl.services.replay.reverb import adders, dataset

from . import spec_utils


@dataclasses.dataclass
class ReverbReplay:
    server: reverb.Server
    adder: adders.Adder
    data_iterator: Iterator[reverb.ReplaySample]
    client: Optional[reverb.Client] = None
    can_sample: Callable[[], bool] = lambda: True


def make_reverb_prioritized_sequence_replay(
    environment_spec: spec_utils.EnvironmentSpec,
    extra_spec: spec_utils.TreeSpec = (),
    batch_size: int = 32,
    max_replay_size: int = 100_000,
    min_replay_size: int = 1,
    priority_exponent: float = 0.0,
    burn_in_length: int = 40,
    sequence_length: int = 80,
    sequence_period: int = 40,
    replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
    prefetch_size: int = 4,
) -> ReverbReplay:
    """Single-process replay for sequence data from an environment spec."""
    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    replay_table = reverb.Table(
        name=replay_table_name,
        sampler=reverb.selectors.Prioritized(priority_exponent),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(min_replay_size),
        signature=adders.SequenceAdder.signature(environment_spec, extra_spec),
    )
    server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f"localhost:{server.port}"
    client = reverb.Client(address)
    sequence_length = burn_in_length + sequence_length + 1
    adder = adders.SequenceAdder(
        client=client,
        period=sequence_period,
        sequence_length=sequence_length,
        delta_encoded=True,
    )

    # The dataset provides an interface to sample from replay.
    data_iterator = dataset.make_reverb_dataset(
        table=replay_table_name,
        server_address=address,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
        environment_spec=environment_spec,
        extra_spec=extra_spec,
        sequence_length=sequence_length,
    ).as_numpy_iterator()
    return ReverbReplay(server, adder, data_iterator, client)
