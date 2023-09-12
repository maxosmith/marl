"""Tests for sequence_adder."""
from absl.testing import absltest, parameterized

from marl.services.replays import end_behavior
from marl.services.replays.reverb.adders import sequence_adder, test_utils

_TEST_CASES = [
    dict(
        testcase_name="basic_api",
        notes="Basic API test.",
        sequence_length=3,
        period=1,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.transition(reward=3.0, observation=3)),
            (0, test_utils.transition(reward=5.0, observation=4)),
            (0, test_utils.termination(reward=7.0, observation=5)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 5.0, False, False, ()),
            ],
            [
                (2, 0, 3.0, False, False, ()),
                (3, 0, 5.0, False, False, ()),
                (4, 0, 7.0, False, False, ()),
            ],
            [
                (3, 0, 5.0, False, False, ()),
                (4, 0, 7.0, False, False, ()),
                (5, 0, 0.0, False, True, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.ZERO_PAD,
    ),
    dict(
        testcase_name="length_one",
        notes="Basic API test for sequence lengths.",
        sequence_length=1,
        period=1,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.transition(reward=3.0, observation=3)),
            (0, test_utils.transition(reward=5.0, observation=4)),
            (0, test_utils.termination(reward=7.0, observation=5)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [(1, 0, 2.0, True, False, ())],
            [(2, 0, 3.0, False, False, ())],
            [(3, 0, 5.0, False, False, ())],
            [(4, 0, 7.0, False, False, ())],
            [(5, 0, 0.0, False, True, ())],
        ),
        end_behavior=end_behavior.EndBehavior.ZERO_PAD,
    ),
    dict(
        testcase_name="period_two",
        notes="Basic API test for sequence periods.",
        sequence_length=3,
        period=2,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.transition(reward=3.0, observation=3)),
            (0, test_utils.transition(reward=5.0, observation=4)),
            (0, test_utils.termination(reward=7.0, observation=5)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 5.0, False, False, ()),
            ],
            [
                (3, 0, 5.0, False, False, ()),
                (4, 0, 7.0, False, False, ()),
                (5, 0, 0.0, False, True, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.ZERO_PAD,
    ),
    dict(
        testcase_name="early_termination_period_one",
        notes="""
          Episode terminates before all sequences can be written with `ZERO_PAD`.

          Episode steps (digits) and writing events (W):
              1 2 3
                  W

          A partial sequence of (2, 3) is never written.
        """,
        sequence_length=3,
        period=1,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 0.0, False, True, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.ZERO_PAD,
    ),
    dict(
        testcase_name="early_termination_period_two",
        notes="""
          Episode terminates before all sequences can be written with `ZERO_PAD`.

          Episode steps (digits) and writing events (W):
              1 2 3
                  W

          A partial sequence of (3) is never written.
        """,
        sequence_length=3,
        period=2,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 0.0, False, True, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.ZERO_PAD,
    ),
    dict(
        testcase_name="early_termination_period_one_multiple_writes",
        notes="""
          Episode terminates before all sequences can be written with `ZERO_PAD`.

          Episode steps (digits) and writing events (W):
              1 2 3 4
                  W W

          A partial sequence of (3, 4) is never written.
        """,
        sequence_length=3,
        period=1,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.transition(reward=4.0, observation=3)),
            (0, test_utils.termination(reward=3.0, observation=4)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 4.0, False, False, ()),
                (3, 0, 3.0, False, False, ()),
            ],
            [
                (2, 0, 4.0, False, False, ()),
                (3, 0, 3.0, False, False, ()),
                (4, 0, 0.0, False, True, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.ZERO_PAD,
    ),
    dict(
        testcase_name="early_termination_period_two_multiple_writes",
        notes="""
          Episode terminates before all sequences can be written with `ZERO_PAD`.

          Episode steps (digits) and writing events (W):
              1 2 3 4 5
                  W   W

          A partial sequence of (5) is never written.
        """,
        sequence_length=3,
        period=2,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.transition(reward=4.0, observation=3)),
            (0, test_utils.transition(reward=4.0, observation=4)),
            (0, test_utils.termination(reward=3.0, observation=5)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 4.0, False, False, ()),
                (3, 0, 4.0, False, False, ()),
            ],
            [
                (3, 0, 4.0, False, False, ()),
                (4, 0, 3.0, False, False, ()),
                (5, 0, 0.0, False, True, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.ZERO_PAD,
    ),
    dict(
        testcase_name="early_termination_padding_period_one",
        notes="Episode terminates before full sequence, requiring padding.",
        sequence_length=4,
        period=1,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 0.0, False, True, ()),
                (0, 0, 0.0, False, False, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.ZERO_PAD,
    ),
    dict(
        testcase_name="early_termination_padding_period_two",
        notes="Episode terminates before full sequence, requiring padding.",
        sequence_length=4,
        period=2,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 0.0, False, True, ()),
                (0, 0, 0.0, False, False, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.ZERO_PAD,
    ),
    dict(
        testcase_name="early_termination_with_truncation",
        notes="Episode ending before sequence, but truncated.",
        sequence_length=4,
        period=1,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 0.0, False, True, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.TRUNCATE,
    ),
    dict(
        testcase_name="long_episode_padding",
        notes="If a period and length match there is no overlap with padding.",
        sequence_length=3,
        period=3,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.transition(reward=3.0, observation=3)),
            (0, test_utils.transition(reward=5.0, observation=4)),
            (0, test_utils.transition(reward=7.0, observation=5)),
            (0, test_utils.transition(reward=9.0, observation=6)),
            (0, test_utils.transition(reward=11.0, observation=7)),
            (0, test_utils.termination(reward=13.0, observation=8)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 5.0, False, False, ()),
            ],
            [
                (4, 0, 7.0, False, False, ()),
                (5, 0, 9.0, False, False, ()),
                (6, 0, 11.0, False, False, ()),
            ],
            [
                (7, 0, 13.0, False, False, ()),
                (8, 0, 0.0, False, True, ()),
                (0, 0, 0.0, False, False, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.ZERO_PAD,
    ),
    dict(
        testcase_name="long_episode_no_padding",
        notes="If a period and length match there is no overlap without padding",
        sequence_length=3,
        period=3,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.transition(reward=3.0, observation=3)),
            (0, test_utils.transition(reward=5.0, observation=4)),
            (0, test_utils.transition(reward=7.0, observation=5)),
            (0, test_utils.transition(reward=9.0, observation=6)),
            (0, test_utils.transition(reward=11.0, observation=7)),
            (0, test_utils.termination(reward=13.0, observation=8)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 5.0, False, False, ()),
            ],
            [
                (4, 0, 7.0, False, False, ()),
                (5, 0, 9.0, False, False, ()),
                (6, 0, 11.0, False, False, ()),
            ],
            [
                (7, 0, 13.0, False, False, ()),
                (8, 0, 0.0, False, True, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.TRUNCATE,
    ),
    dict(
        testcase_name="end_behavior_WRITE",
        notes="""
          End behavior WRITE forces a complete sequence with larger period.

          Episode steps (digits) and writing events (W):
              1 2 3 4 5 6
                  W   W W

              1 2 3
              . . 3 4 5
              . . . 4 5 6
        """,
        sequence_length=3,
        period=2,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.transition(reward=3.0, observation=3)),
            (0, test_utils.transition(reward=5.0, observation=4)),
            (0, test_utils.transition(reward=7.0, observation=5)),
            (0, test_utils.termination(reward=8.0, observation=6)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 5.0, False, False, ()),
            ],
            [
                (3, 0, 5.0, False, False, ()),
                (4, 0, 7.0, False, False, ()),
                (5, 0, 8.0, False, False, ()),
            ],
            [
                (4, 0, 7.0, False, False, ()),
                (5, 0, 8.0, False, False, ()),
                (6, 0, 0.0, False, True, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.WRITE,
    ),
    dict(
        testcase_name="non_breaking_sequence_on_episode_reset",
        notes="End behavior continue doesn't write partial sequences at end of episode.",
        sequence_length=3,
        period=2,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.transition(reward=3.0, observation=3)),
            (0, test_utils.transition(reward=5.0, observation=4)),
            (0, test_utils.transition(reward=7.0, observation=5)),
            (0, test_utils.transition(reward=9.0, observation=6)),
            (0, test_utils.transition(reward=11.0, observation=7)),
            (0, test_utils.termination(reward=13.0, observation=8)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 5.0, False, False, ()),
            ],
            [
                (3, 0, 5.0, False, False, ()),
                (4, 0, 7.0, False, False, ()),
                (5, 0, 9.0, False, False, ()),
            ],
            [
                (5, 0, 9.0, False, False, ()),
                (6, 0, 11.0, False, False, ()),
                (7, 0, 13.0, False, False, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.CONTINUE,
    ),
    dict(
        testcase_name="non_breaking_sequence_multiple_terminated_episodes",
        notes="Continue across multiple episodes.",
        sequence_length=3,
        period=2,
        steps=(
            (0, test_utils.restart(1)),
            (0, test_utils.transition(reward=2.0, observation=2)),
            (0, test_utils.transition(reward=3.0, observation=3)),
            (0, test_utils.transition(reward=5.0, observation=4)),
            (0, test_utils.transition(reward=7.0, observation=5)),
            (0, test_utils.transition(reward=9.0, observation=6)),
            (0, test_utils.termination(reward=13.0, observation=7)),
        ),
        expected_sequences=(
            # (observation, action, reward, start_of_episode, end_of_episode, extra)
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 5.0, False, False, ()),
            ],
            [
                (3, 0, 5.0, False, False, ()),
                (4, 0, 7.0, False, False, ()),
                (5, 0, 9.0, False, False, ()),
            ],
            [
                (5, 0, 9.0, False, False, ()),
                (6, 0, 13.0, False, False, ()),
                (7, 0, 0.0, False, True, ()),
            ],
            [
                (7, 0, 0.0, False, True, ()),
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
            ],
            [
                (2, 0, 3.0, False, False, ()),
                (3, 0, 5.0, False, False, ()),
                (4, 0, 7.0, False, False, ()),
            ],
            [
                (4, 0, 7.0, False, False, ()),
                (5, 0, 9.0, False, False, ()),
                (6, 0, 13.0, False, False, ()),
            ],
            [
                (6, 0, 13.0, False, False, ()),
                (7, 0, 0.0, False, True, ()),
                (1, 0, 2.0, True, False, ()),
            ],
            [
                (1, 0, 2.0, True, False, ()),
                (2, 0, 3.0, False, False, ()),
                (3, 0, 5.0, False, False, ()),
            ],
            [
                (3, 0, 5.0, False, False, ()),
                (4, 0, 7.0, False, False, ()),
                (5, 0, 9.0, False, False, ()),
            ],
            [
                (5, 0, 9.0, False, False, ()),
                (6, 0, 13.0, False, False, ()),
                (7, 0, 0.0, False, True, ()),
            ],
        ),
        end_behavior=end_behavior.EndBehavior.CONTINUE,
        repeat_episode_times=3,
    ),
]


class SequenceAdderTest(test_utils.ReverbAdderTestMixin, parameterized.TestCase):

  @parameterized.named_parameters(_TEST_CASES)
  def test_adder(
      self,
      notes: str,
      sequence_length: int,
      period: int,
      steps,
      expected_sequences,
      end_behavior: end_behavior.EndBehavior = end_behavior.EndBehavior.ZERO_PAD,
      repeat_episode_times: int = 1,
  ):
    del notes  # Used for extended documentation of test case purpose.
    adder = sequence_adder.SequenceAdder(
        self.client,
        sequence_length=sequence_length,
        period=period,
        end_of_episode_behavior=end_behavior,
    )
    super().run_test_adder(
        adder=adder,
        steps=steps,
        expected_items=expected_sequences,
        repeat_episode_times=repeat_episode_times,
        end_behavior=end_behavior,
        signature=adder.signature(*test_utils.get_specs(steps[0])),
    )

  @parameterized.parameters(
      (True, True, end_behavior.EndBehavior.ZERO_PAD),
      (False, True, end_behavior.EndBehavior.TRUNCATE),
      (False, False, end_behavior.EndBehavior.CONTINUE),
  )
  def test_end_of_episode_behavior_set_correctly(
      self, pad_end_of_episode, break_end_of_episode, expected_behavior
  ):
    adder = sequence_adder.SequenceAdder(
        self.client,
        sequence_length=5,
        period=3,
        pad_end_of_episode=pad_end_of_episode,
        break_end_of_episode=break_end_of_episode,
    )
    self.assertEqual(adder._end_of_episode_behavior, expected_behavior)


if __name__ == "__main__":
  absltest.main()
