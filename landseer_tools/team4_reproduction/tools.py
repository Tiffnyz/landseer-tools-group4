"""LangChain wrapper for Validate_metric. All logic lives in core.py."""

from __future__ import annotations

from langchain.tools import tool
from pydantic import BaseModel, Field

from .core import validate_entry
from .record_store import get_record_store, resolve_metric_id
from .schemas import ValidationResult


class ValidateMetricInput(BaseModel):
    metric: str = Field(
        description=("Metric id from the metrics record, e.g. 'fingerprint_auc@cifar10'. "
                     "A bare name like 'fingerprint_auc' works if only one setting has it."))


def run_validate_metric(metric: str, write_back: bool = True) -> dict:
    """Plain-function version of the tool (use this in tests and scripts)."""
    try:
        store = get_record_store()
        ids = store.list_metric_ids()
    except Exception as e:  # never raise into the agent loop
        return ValidationResult(metric_id=None, verdict="not_found",
                                message=f"Could not read the metrics record: {e}").model_dump(mode="json")

    metric_id, candidates = resolve_metric_id(metric, ids)
    if metric_id is None:
        what = "matches more than one metric" if candidates and len(candidates) < len(ids) else "is not in the record"
        return ValidationResult(
            metric_id=None, verdict="not_found", candidates=candidates,
            message=f"'{metric}' {what}. Call validate_metric again with one of: {', '.join(candidates) or '(record is empty)'}",
        ).model_dump(mode="json")

    result = validate_entry(store.get_entry(metric_id))
    if write_back and result.verdict not in ("not_found",):
        try:
            store.save_validation(metric_id, result)
        except Exception as e:
            result.warnings.append(f"Verdict was not saved to the record: {e}")
    return result.model_dump(mode="json")


@tool("validate_metric", args_schema=ValidateMetricInput)
def validate_metric(metric: str) -> dict:
    """Compare a reproduced metric with the value the paper reported, using the metrics record.

    Call this after Reproduce_metric has produced a value and it has been saved with
    Insert_into_record. Returns a verdict:
    - reproduced: within tolerance (default 3 points) of the paper's value. Done.
    - out_of_tolerance: worse than the paper. Check setting, hyperparameters, seeds,
      and whether training finished; fix minimally and reproduce again.
    - better_than_reported: suspiciously better. Check for evaluation leakage or a
      different test split before accepting.
    - not_comparable: the run used a different model/dataset (a replication) or the
      value's scale looks wrong (e.g. 85 vs 0.85). Follow the message.
    - not_run / invalid / not_found: no usable value, or a bad metric name. Follow the message.
    The verdict is saved back into the record.
    """
    return run_validate_metric(metric)
