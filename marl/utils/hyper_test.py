"""Test cases for the `hyper` module."""
import dataclasses

from absl.testing import absltest, parameterized

from marl.utils import hyper


@dataclasses.dataclass
class _Config0:
    a: float = 0.003
    b: int = 64


class HyperTest(parameterized.TestCase):
    @parameterized.parameters(
        [
            dict(config_type=_Config0, sweep=dict(a=0.02, b=32)),
        ]
    )
    def test_cast(self, config_type, sweep):
        casted = hyper.cast(config_type, [sweep])[0]
        assert isinstance(casted, config_type)
        for key, value in sweep.items():
            self.assertEqual(getattr(casted, key), value)

    @parameterized.parameters(
        [
            dict(base=_Config0(), override=dict()),
            dict(base=_Config0(), override=dict(a=0.2)),
            dict(base=_Config0(), override=dict(b=3)),
            dict(base=_Config0(), override=dict(a=0.2, b=3)),
        ]
    )
    def test_default(self, base, override):
        result = hyper.default(base, [override])[0]
        for key, value in result.items():
            if key in override:
                self.assertEqual(override[key], value)
            else:
                self.assertEqual(base[key], value)


if __name__ == "__main__":
    absltest.main()
