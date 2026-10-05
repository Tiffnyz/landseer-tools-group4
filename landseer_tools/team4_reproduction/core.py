"""Comparison logic for Validate_metric. Plain functions, no LangChain, no I/O."""

from __future__ import annotations

import math
import re
import statistics
from typing import Optional

from .schemas import (
    Direction,
    MetricEntry,
    Scale,
    Setting,
    Tolerance,
    ValidationResult,
)

DEFAULT_TOLERANCE = 0.03  # Landseer Sec. 6.2.2


# ---------------------------------------------------------------- helpers

def _norm(s: object) -> str:
    """'ResNet-20' == 'resnet20' == 'RESNET_20'."""
    return re.sub(r"[^a-z0-9.]", "", str(s).lower())


def to_comparable(value: float, scale: Scale) -> float:
    """Put fraction and percent values on one scale (fraction). RAW is unchanged."""
    return value / 100.0 if scale == Scale.PERCENT else value


def default_tolerance(scale: Scale) -> Tolerance:
    if scale == Scale.RAW:
        return Tolerance(type="relative", value=DEFAULT_TOLERANCE)
    return Tolerance(type="absolute", value=DEFAULT_TOLERANCE)


def setting_mismatches(reported: Setting, run: Optional[Setting]) -> list[str]:
    """Fields the paper specified that the run did differently. Empty = same setting."""
    if run is None:
        return []
    diffs: list[str] = []
    for field in ("model", "dataset"):
        want, got = getattr(reported, field), getattr(run, field)
        if want is not None and got is not None and _norm(want) != _norm(got):
            diffs.append(f"{field}: paper={want!r}, run={got!r}")
    for key, want in reported.extra.items():
        if key in run.extra and _norm(run.extra[key]) != _norm(want):
            diffs.append(f"{key}: paper={want!r}, run={run.extra[key]!r}")
    return diffs


def _looks_like_scale_error(value: float, scale: Scale) -> Optional[str]:
    if scale == Scale.FRACTION and 1.0 < value <= 100.0:
        return (f"value {value} is above 1 but the scale is 'fraction'; "
                "it is probably a percentage. Re-insert it with scale='percent'.")
    if scale == Scale.PERCENT and 0.0 < value <= 1.0:
        return (f"value {value} is at most 1 but the scale is 'percent'; "
                "it may be a fraction. Check and re-insert with the right scale.")
    return None


# ---------------------------------------------------------------- main entry

def validate_entry(entry: MetricEntry) -> ValidationResult:
    """Judge the reproduced runs in `entry` against its reported value."""
    base = dict(metric_id=entry.metric_id, reported=entry.reported_value)
    warnings: list[str] = []

    if not math.isfinite(entry.reported_value):
        return ValidationResult(**base, verdict="invalid",
                                message="The reported value in the record is not a finite number; fix the record.")

    if not entry.reproduced:
        return ValidationResult(**base, verdict="not_run",
                                message=(f"No reproduced value for '{entry.metric_id}' yet. Run Reproduce_metric, "
                                         "insert the value with Insert_into_record, then validate again."))

    # 1. Split runs into same-setting (reproduction) vs different-setting (replication).
    same, other = [], []
    for run in entry.reproduced:
        diffs = setting_mismatches(entry.setting, run.setting)
        (other if diffs else same).append((run, diffs))

    if not same:
        detail = "; ".join(sorted({d for _, ds in other for d in ds}))
        return ValidationResult(
            **base, verdict="not_comparable", excluded_runs=len(other),
            message=("All reproduced runs used a different setting from the paper "
                     f"({detail}). That is a replication, not a reproduction, so it can't be judged "
                     "against the paper's value. Re-run in the paper's setting to validate."))
    if other:
        warnings.append(f"Ignored {len(other)} run(s) with a different setting from the paper.")

    # 2. Check values and scales.
    values: list[float] = []
    for run, _ in same:
        scale = run.scale or entry.scale
        if not math.isfinite(run.value):
            return ValidationResult(**base, verdict="invalid", n_runs=len(same),
                                    message=f"A reproduced value is {run.value}; the run likely crashed or diverged. Check the logs.")
        if (entry.scale == Scale.RAW) != (scale == Scale.RAW):
            return ValidationResult(**base, verdict="not_comparable", n_runs=len(same),
                                    message="One value is on the 'raw' scale and the other is not, so they can't be compared.")
        problem = _looks_like_scale_error(run.value, scale)
        if problem:
            return ValidationResult(**base, verdict="not_comparable", n_runs=len(same), message=problem)
        values.append(to_comparable(run.value, scale))

    ref_problem = _looks_like_scale_error(entry.reported_value, entry.scale)
    if ref_problem:
        return ValidationResult(**base, verdict="not_comparable", n_runs=len(values),
                                message="Reported value in the record: " + ref_problem)

    # 3. Aggregate and compare.
    reported = to_comparable(entry.reported_value, entry.scale)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    delta = mean - reported

    tol = entry.tolerance or default_tolerance(entry.scale)
    if tol.type == "relative":
        if reported == 0:
            band = tol.value
            warnings.append("Reported value is 0, so relative tolerance was applied as absolute.")
        else:
            band = tol.value * abs(reported)
    else:
        band = tol.value

    # Small epsilon so 0.97 vs 1.00 with tol 0.03 isn't failed by float rounding.
    within = abs(delta) <= band + 1e-9

    if std is not None and std > band:
        warnings.append(f"Run-to-run std ({std:.4g}) is larger than the tolerance ({band:.4g}); "
                        "the verdict is unreliable. Consider more seeds.")
    if len(values) == 1:
        warnings.append("Only one run; a second seed would make the verdict more trustworthy.")
    if any(abs(v - reported) < 1e-12 for v in values) and reported not in (0.0, 1.0):
        warnings.append("A reproduced value exactly equals the reported value; check it was "
                        "measured and not copied from the paper.")

    code_modified = any(run.code_modified for run, _ in same)
    pts = "" if entry.scale == Scale.RAW else " (fraction scale)"
    summary = (f"reproduced {mean:.4g} vs reported {reported:.4g}{pts}, "
               f"delta {delta:+.4g}, tolerance ±{band:.4g} ({tol.type}).")

    if within:
        verdict = "reproduced"
        how = "with minimal code modifications" if code_modified else "with the original code"
        message = f"Reproduced {how}: {summary}"
    elif (entry.direction == Direction.HIGHER and delta > 0) or (entry.direction == Direction.LOWER and delta < 0):
        verdict = "better_than_reported"
        message = ("Outside tolerance but better than the paper: " + summary +
                   " Check for evaluation leakage or a setting mismatch (e.g. a different test split) before accepting.")
    else:
        verdict = "out_of_tolerance"
        message = ("Not reproduced: " + summary +
                   " Check the setting, hyperparameters, seeds, and whether training ran to completion.")

    return ValidationResult(
        **base,
        verdict=verdict,
        message=message,
        reproduced_mean=round(mean, 6),
        reproduced_std=None if std is None else round(std, 6),
        n_runs=len(values),
        delta=round(delta, 6),
        tolerance_used=band,
        tolerance_type=tol.type,
        scale_compared_on="raw" if entry.scale == Scale.RAW else "fraction",
        code_modified=code_modified,
        excluded_runs=len(other),
        warnings=warnings,
    )
