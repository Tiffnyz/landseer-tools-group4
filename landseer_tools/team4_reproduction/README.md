# Team 4: `validate_metric`

Compares a reproduced metric with the value the paper reported, using the
metrics record from Team 2, and returns a verdict the Landseer agent can act on.

```
landseer_tools/team4_reproduction/
  schemas.py        record + result models (our proposal for Team 2's record format)
  core.py           comparison logic: plain functions, no LangChain, no I/O
  record_store.py   reads/writes the record; swap in Team 2's store here
  tools.py          LangChain @tool wrapper (name: validate_metric)
  examples/uap_fingerprinting_record.json
  tests/            31 tests: pytest landseer_tools/team4_reproduction/tests
```

## Use it

```python
from langchain.agents import create_agent
from landseer_tools.team4_reproduction import validate_metric, JsonRecordStore, set_record_store

set_record_store(JsonRecordStore("records/uap_fingerprinting.json"))  # or set LANDSEER_RECORD_PATH
agent = create_agent(model="...", tools=[..., validate_metric])
```

Without an agent: `run_validate_metric("fingerprint_auc@cifar10")`.

## Verdicts

| Verdict | Meaning | What the agent should do |
|---|---|---|
| `reproduced` | Within tolerance of the paper | Done |
| `out_of_tolerance` | Worse than the paper | Check setting, hyperparameters, seeds; fix minimally, re-run |
| `better_than_reported` | Outside tolerance, in the good direction | Check for leakage or a different test split |
| `not_comparable` | Different model/dataset (a replication), or scale mix-up (85 vs 0.85) | Follow the message |
| `not_run` | No reproduced value yet | Reproduce_metric, then Insert_into_record |
| `invalid` | NaN/inf | Run crashed or diverged; check logs |
| `not_found` | Unknown or ambiguous metric name | Retry with one of `candidates` |

The verdict is written back to the entry's `validation` field.

## Decisions to confirm with the professor / other teams

1. **3% tolerance = 3 points absolute** for fraction/percent metrics, **3% relative**
   for raw metrics (e.g. DP epsilon). The Landseer paper doesn't say which. Override
   per metric with `"tolerance": {"type": "relative", "value": 0.03}`.
2. **Reproduction vs replication.** Runs whose model/dataset/extra settings differ from
   the paper's are excluded. If none match, the verdict is `not_comparable`. This
   matters once Team 3's Add_model / Add_dataset change the setting.
3. **Mean over seeds** is compared to the reported value; a warning is added if the
   spread is bigger than the tolerance.
4. **Scale mix-ups are reported, not auto-fixed**, so a wrong number never passes silently.

## Fields we need in Team 2's record

`metric_id`, `reported_value`, `scale`, `direction`, `setting` (model, dataset, extra),
optional `tolerance`, a list of `reproduced` runs (value, seed, scale, setting,
code_modified), and a `validation` field we can write to. See `schemas.py`.
If their format differs, write an adapter class with `list_metric_ids`,
`get_entry` and `save_validation` and pass it to `set_record_store()`.

The reproduced values in `examples/` are made up for testing, not real results.
