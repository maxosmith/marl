"""Available optiosn for writing behavior at episode ends."""
import enum


class EndBehavior(enum.Enum):
  """Class to enumerate available options for writing behavior at episode ends.

  Example:
    sequence_length = 3
    period = 2

  Episode steps (digits) and writing events (W):
             1 2 3 4 5 6
                 W   W

  First two sequences:
             1 2 3
             . . 3 4 5

  Written sequences for the different end of episode behaviors:

  Here are the last written sequences for each end of episode behavior:
   WRITE     . . . 4 5 6
   CONTINUE  . . . . 5 6 F
   ZERO_PAD  . . . . 5 6 0
   TRUNCATE  . . . . 5 6

  Key:
    F: First step of the next episode
    0: Zero-filled Step
  """

  WRITE = "write_buffer"
  CONTINUE = "continue_to_next_episode"
  ZERO_PAD = "zero_pad_til_next_write"
  TRUNCATE = "write_truncated_buffer"
