import os
from typing import Optional

import pandas as pd
from tensorflow.python.framework import tensor_util
from tensorflow.python.summary.summary_iterator import summary_iterator


def events_to_dataframe(root_dir: str, sort_by: Optional[str] = None) -> pd.DataFrame:
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses all events data.
    If the `sort_by` value is provided then it will use that column to sort values;
    typically `wall_time` or `step`.

    NOTE: this converts all data, and may take a long time to return.

    Paramters:
        root_dir: path to root dir with tensorboard data.
        sort_by: c olumn name to sort by.

    Returns:
        DataFrame containing the tensorboard data.
    """

    def convert_tfevent(filepath):
        return pd.DataFrame([parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)])

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=tensor_util.MakeNdarray(tfevent.summary.value[0].tensor),
        )

    columns_order = ["wall_time", "name", "step", "value"]

    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    df = all_df.reset_index(drop=True)

    # Pivot the table so that the all logged values are a column.
    df = pd.pivot(df, index="step", columns="name", values=["value", "wall_time"]).reset_index()
    wall_time = df[("wall_time", df["wall_time"].columns[0])]
    del df["wall_time"]
    df[("step", "step")] = df["step"]
    del df[("step", "")]
    df[("wall_time", "wall_time")] = wall_time
    df.columns = df.columns.get_level_values(1)
    return df
