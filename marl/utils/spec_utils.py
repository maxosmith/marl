"""Specification (spec) utility functions."""
import numpy as np
import tree

from marl import specs, types


def spec_like(x: types.Tree) -> specs.TreeSpec:
  """Creates a specificaiton for the input.

  Args:
    x: Data to spec.

  Returns:
    Specification for the shape and typing of x.
  """

  def _build_spec(x: types.Tree) -> specs.TreeSpec:
    x = np.asarray(x)
    return specs.ArraySpec(x.shape, x.dtype)

  return tree.map_structure(_build_spec, x)


def zeros_like(spec: specs.TreeSpec) -> types.Tree:
  """Generate an instance of a spec containing all zeros."""
  return tree.map_structure(lambda x: np.zeros(x.shape, x.dtype), spec)
