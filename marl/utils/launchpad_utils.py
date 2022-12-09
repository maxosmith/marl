"""Utility functions for operating with launchpad programs."""
import os.path as osp
import re

# 7-bit C1 ANSI sequences
_ANSI_ESCAPE = re.compile(
    r"""
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by a control sequence
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
""",
    re.VERBOSE,
)


def split_log(path: str, output_dir: str):
    """Split a log containing all node outputs into per-node outputs.

    NOTE: This function makes assumptions about the log structure:
        * No nodes are named `main`.
        * Only output from nodes contain matching brackets.
    """
    destinations = {}

    with open(path, "r") as source:
        while line := source.readline().rstrip():
            line = _ANSI_ESCAPE.sub("", line)

            # Lines are written as:
            #   [group_name/index] Output text maybe including more ] throughout.
            # Lines are written with colour, meaning the true string repr is:
            #   \\1ab[1;32m[group_name/index] Output text maybe including more ] throughout.\\x1b[0;0m]

            if ("[" not in line) or ("]" not in line):
                # Assumed that this line not from a launchpad node.
                group_name = "main"

            else:
                # Parse out the group name and index.
                bracket_index = line.index("]")

                group_name = line[1:bracket_index]  # [group_name/index] ... -> group_name/index
                group_name = group_name.replace("/", "_")

            # Write the line to a node-specific log file.
            if group_name not in destinations:
                destinations[group_name] = open(osp.join(output_dir, f"{group_name}.log"), "w")
            destinations[group_name].write(f"{line}\n")
