"""Data models for the metrics record and validation results.

The metrics record is owned by Team 2 (Initialize_metrics_record / Insert_into_record).
These models are Team 4's proposal for the fields Validate_metric needs. If Team 2's
final format differs, only `record_store.py` needs an adapter; the comparison logic
in `core.py` works on these models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Scale(str, Enum):
    """How a metric value is expressed."""

    FRACTION = "fraction"  # 0..1, e.g. AUC = 0.98, accuracy = 0.85
    PERCENT = "percent"    # 0..100, e.g. accuracy = 85.0
    RAW = "raw"            # unbounded, e.g. DP epsilon, explanation error


class Direction(str, Enum):
    """Which way is better for this metric (Landseer Table 1 up/down arrows)."""

    HIGHER = "higher"
    LOWER = "lower"
    NONE = "none"  # no preferred direction; any deviation is just a deviation


class Tolerance(BaseModel):
    """Allowed deviation from the reported value.

    Landseer (Sec. 6.2.2) treats a defense as reproduced if it lands within 3% of
    the reported value but does not say whether that is absolute or relative.
    Default here: 3 percentage points (absolute, on the fraction scale) for
    fraction/percent metrics. Use relative tolerance for RAW metrics.
    """

    type: Literal["absolute", "relative"] = "absolute"
    value: float = Field(0.03, ge=0, description="0.03 = 3 points (absolute) or 3% (relative)")


class Setting(BaseModel):
    """The experimental setting a value belongs to.

    Only fields the paper reports are compared. Changing the model or dataset
    (Team 3's Add_model / Add_dataset) turns a reproduction into a replication,
    which must not be judged against the paper's number.
    """

    model: Optional[str] = None
    dataset: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict, description="Other settings, e.g. epsilon, n_fingerprints")


class ReproducedRun(BaseModel):
    """One reproduced value, as written by Insert_into_record."""

    value: float
    seed: Optional[int] = None
    scale: Optional[Scale] = None  # defaults to the entry's scale
    setting: Optional[Setting] = None  # defaults to the entry's reported setting
    code_modified: bool = False  # True if the agent patched the repo (minimal changes)
    notes: Optional[str] = None


class MetricEntry(BaseModel):
    """One metric in the record: what the paper reported plus what we reproduced."""

    metric_id: str = Field(description="Unique id, e.g. 'fingerprint_auc@cifar10'")
    name: str = Field(description="Metric name, e.g. 'fingerprint_auc'")
    description: Optional[str] = None
    source: Optional[str] = Field(None, description="Where in the paper, e.g. 'Sec. 5, AUC table'")

    reported_value: float
    reported_std: Optional[float] = None
    scale: Scale = Scale.FRACTION
    direction: Direction = Direction.HIGHER
    tolerance: Optional[Tolerance] = None  # None -> default for the scale
    setting: Setting = Field(default_factory=Setting)

    reproduced: list[ReproducedRun] = Field(default_factory=list)
    validation: Optional[dict[str, Any]] = None  # last ValidationResult, written back


class MetricsRecord(BaseModel):
    """The whole record for one paper/tool."""

    paper: Optional[str] = None
    code_url: Optional[str] = None
    metrics: list[MetricEntry] = Field(default_factory=list)


Verdict = Literal[
    "reproduced",            # within tolerance of the reported value
    "better_than_reported",  # outside tolerance, in the good direction (worth a look)
    "out_of_tolerance",      # outside tolerance, in the bad direction
    "not_comparable",        # setting or scale mismatch; can't judge against the paper
    "not_run",               # no reproduced value in the record yet
    "invalid",               # NaN/inf or otherwise unusable values
    "not_found",             # metric isn't in the record (or is ambiguous)
]


class ValidationResult(BaseModel):
    """What Validate_metric returns. `message` is written for the agent to read."""

    metric_id: Optional[str]
    verdict: Verdict
    message: str
    reported: Optional[float] = None
    reproduced_mean: Optional[float] = None
    reproduced_std: Optional[float] = None
    n_runs: int = 0
    delta: Optional[float] = None
    tolerance_used: Optional[float] = None
    tolerance_type: Optional[str] = None
    scale_compared_on: Optional[str] = None
    code_modified: Optional[bool] = None
    excluded_runs: int = 0
    warnings: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
