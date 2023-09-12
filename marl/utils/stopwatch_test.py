"""Tests for `stopwatch`."""
from absl.testing import absltest, parameterized

from marl.utils import stopwatch


class StopwatchTest(parameterized.TestCase):

  def test_start_stop(self):
    """Tests starting and stopping the stopwatch."""
    watch = stopwatch.Stopwatch()

    for size in range(3):
      watch.start("foo")
      watch.stop("foo")

      splits = watch.get_splits()
      self.assertLen(splits, 1)
      self.assertLen(splits["foo"], size + 1)

  def test_parallel_keys(self):
    """Tests several keys being run in parallel."""
    watch = stopwatch.Stopwatch()

    watch.start("foo")
    watch.start("bar")
    watch.start("buzz")
    watch.stop("bar")
    watch.start("bar")
    watch.stop("bar")
    watch.stop("buzz")
    watch.stop("foo")

    splits = watch.get_splits()
    self.assertLen(splits, 3)
    self.assertLen(splits["foo"], 1)
    self.assertLen(splits["bar"], 2)
    self.assertLen(splits["buzz"], 1)

  def test_split(self):
    """Tests split."""
    watch = stopwatch.Stopwatch()

    watch.start("foo")

    for size in range(3):
      watch.split("foo")

      splits = watch.get_splits()
      self.assertLen(splits, 1)
      self.assertLen(splits["foo"], size + 1)

  def test_clear(self):
    """Tests clear."""
    watch = stopwatch.Stopwatch()

    watch.start("foo")
    watch.stop("foo")
    watch.start("bar")
    watch.stop("bar")
    splits = watch.get_splits()
    self.assertLen(splits, 2)

    watch.clear("foo")
    splits = watch.get_splits()
    self.assertLen(splits, 1)
    self.assertIn("bar", splits)

    watch.start("buzz")
    watch.stop("buzz")
    watch.start("meow")
    watch.stop("meow")
    splits = watch.get_splits()
    self.assertLen(splits, 3)
    self.assertIn("bar", splits)
    self.assertIn("buzz", splits)
    self.assertIn("meow", splits)

    watch.clear(["bar", "buzz"])
    splits = watch.get_splits()
    self.assertLen(splits, 1)
    self.assertIn("meow", splits)

    watch.start("woof")
    watch.stop("woof")
    watch.start("last")
    watch.stop("last")
    splits = watch.get_splits()
    self.assertLen(splits, 3)
    watch.clear()
    splits = watch.get_splits()
    self.assertEmpty(splits)

    watch.start("foo")
    splits = watch.get_splits()
    self.assertLen(splits, 1)
    self.assertIn("foo", splits)
    watch.clear()
    splits = watch.get_splits()
    self.assertEmpty(splits)


if __name__ == "__main__":
  absltest.main()
