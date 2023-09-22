"""Utility functions for launchpad programs."""
import os.path as osp
import pathlib
import re

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


def split_log(path: str | pathlib.Path, output_dir: str | pathlib.Path):
  """Split a log containing logs from all nodes into per-node logs.

  Args:
      path: Path to the log file that needs to be split.
      output_dir: Directory where the split logs will be saved.

  NOTE: This function makes assumptions about the log structure:
      * No nodes are named `main`.
      * Only output from nodes contain matching brackets.
  """
  destinations = {}

  with open(path, "r") as source:
    while line := source.readline().rstrip():
      line = _ANSI_ESCAPE.sub("", line)

      if ("[" not in line) or ("]" not in line):
        group_name = "main"
      else:
        bracket_index = line.index("]")
        group_name = line[1:bracket_index]
        group_name = group_name.replace("/", "_")

      if group_name not in destinations:
        destinations[group_name] = open(osp.join(output_dir, f"{group_name}.log"), "w")
      destinations[group_name].write(f"{line}\n")

  for file in destinations.values():
    file.close()
