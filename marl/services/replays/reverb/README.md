# Reverb Replay Buffer
Launchpad provides an implementation of a Reverb Replay Buffer service.

```python
import launchpad as lp

lp.ReverbNode(
  priority_tables_fn=...,
  checkpoint_ctor=...,
  checkpoint_time_delta_minutes=None,  # Manual checkpointing required.
)
```

