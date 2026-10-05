import json
import math
import shutil
from pathlib import Path

import pytest

from landseer_tools.team4_reproduction.core import validate_entry
from landseer_tools.team4_reproduction.record_store import (
    JsonRecordStore,
    resolve_metric_id,
    set_record_store,
)
from landseer_tools.team4_reproduction.schemas import MetricEntry
from landseer_tools.team4_reproduction.tools import run_validate_metric, validate_metric

EXAMPLE = Path(__file__).parent.parent / "examples" / "uap_fingerprinting_record.json"


def entry(reported=0.85, runs=(), **kw):
    """Build a MetricEntry; runs are floats or dicts."""
    reproduced = [r if isinstance(r, dict) else {"value": r} for r in runs]
    return MetricEntry(metric_id=kw.pop("metric_id", "acc@cifar10"), name="acc",
                       reported_value=reported, reproduced=reproduced, **kw)


# ---------------------------------------------------------------- verdicts

def test_within_tolerance_is_reproduced():
    r = validate_entry(entry(0.85, [0.83, 0.84]))
    assert r.verdict == "reproduced"
    assert r.n_runs == 2 and r.reproduced_mean == pytest.approx(0.835)


def test_exactly_at_tolerance_boundary_passes():
    # 1.00 reported, 0.97 reproduced, tolerance 0.03: float rounding must not fail this
    assert validate_entry(entry(1.0, [0.97])).verdict == "reproduced"


def test_just_outside_tolerance_fails():
    assert validate_entry(entry(1.0, [0.969])).verdict == "out_of_tolerance"


def test_better_than_reported_higher_is_better():
    r = validate_entry(entry(0.80, [0.90]))
    assert r.verdict == "better_than_reported"
    assert "leakage" in r.message


def test_lower_is_better_direction():
    # attack success rate: lower is better
    assert validate_entry(entry(0.20, [0.10], direction="lower")).verdict == "better_than_reported"
    assert validate_entry(entry(0.20, [0.30], direction="lower")).verdict == "out_of_tolerance"


def test_no_direction_never_says_better():
    assert validate_entry(entry(0.5, [0.6], direction="none")).verdict == "out_of_tolerance"


def test_relative_tolerance_on_raw_scale():
    # DP epsilon 50, relative 3% -> band 1.5
    assert validate_entry(entry(50.0, [51.4], scale="raw", direction="lower")).verdict == "reproduced"
    r = validate_entry(entry(50.0, [52.0], scale="raw", direction="lower"))
    assert r.verdict == "out_of_tolerance" and r.tolerance_used == pytest.approx(1.5)


def test_custom_tolerance():
    e = entry(0.85, [0.80], tolerance={"type": "absolute", "value": 0.06})
    assert validate_entry(e).verdict == "reproduced"


def test_relative_tolerance_with_zero_reported_falls_back():
    r = validate_entry(entry(0.0, [0.01], scale="raw", direction="none"))
    assert r.verdict == "reproduced" and r.warnings


# ---------------------------------------------------------------- scales

def test_percent_run_against_fraction_record():
    r = validate_entry(entry(0.85, [{"value": 84.0, "scale": "percent"}]))
    assert r.verdict == "reproduced" and r.reproduced_mean == pytest.approx(0.84)


def test_percent_record_against_fraction_run():
    r = validate_entry(entry(100.0, [{"value": 0.99, "scale": "fraction"}], scale="percent"))
    assert r.verdict == "reproduced"


def test_scale_mixup_is_caught_not_failed():
    # agent inserted 85 but record says fraction: should not be "out_of_tolerance"
    r = validate_entry(entry(0.85, [85.0]))
    assert r.verdict == "not_comparable" and "percent" in r.message


def test_raw_vs_bounded_not_comparable():
    r = validate_entry(entry(0.85, [{"value": 0.85, "scale": "raw"}]))
    assert r.verdict == "not_comparable"


# ---------------------------------------------------------------- settings

def test_different_dataset_is_replication_not_reproduction():
    e = entry(0.85, [{"value": 0.60, "setting": {"dataset": "cifar100"}}], setting={"dataset": "cifar10"})
    r = validate_entry(e)
    assert r.verdict == "not_comparable" and "replication" in r.message


def test_setting_names_are_normalized():
    e = entry(0.85, [{"value": 0.85, "setting": {"model": "resnet20", "dataset": "CIFAR10"}}],
              setting={"model": "ResNet-20", "dataset": "cifar-10"})
    assert validate_entry(e).verdict == "reproduced"


def test_mixed_settings_only_counts_matching_runs():
    e = entry(0.85, [{"value": 0.85}, {"value": 0.10, "setting": {"dataset": "svhn"}}],
              setting={"dataset": "cifar10"})
    r = validate_entry(e)
    assert r.verdict == "reproduced" and r.excluded_runs == 1 and r.n_runs == 1


def test_extra_setting_mismatch():
    e = entry(0.85, [{"value": 0.85, "setting": {"extra": {"n_fingerprints": 20}}}],
              setting={"extra": {"n_fingerprints": 100}})
    assert validate_entry(e).verdict == "not_comparable"


# ---------------------------------------------------------------- bad input & warnings

def test_no_runs():
    assert validate_entry(entry(0.85, [])).verdict == "not_run"


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nan_value_is_invalid(bad):
    assert validate_entry(entry(0.85, [bad])).verdict == "invalid"


def test_noisy_runs_warn():
    r = validate_entry(entry(0.50, [0.40, 0.60]))
    assert any("std" in w for w in r.warnings)


def test_single_run_warns():
    assert any("one run" in w for w in validate_entry(entry(0.85, [0.85])).warnings)


def test_copied_value_warns():
    assert any("copied" in w for w in validate_entry(entry(0.8512, [0.8512, 0.85])).warnings)


def test_code_modified_reported():
    r = validate_entry(entry(0.85, [{"value": 0.85, "code_modified": True}]))
    assert r.code_modified and "minimal code modifications" in r.message


# ---------------------------------------------------------------- id resolution

IDS = ["fingerprint_auc@cifar10", "fingerprint_auc@fmnist", "detection_rate@cifar10"]


def test_resolve_exact_and_unique_name():
    assert resolve_metric_id("fingerprint_auc@cifar10", IDS) == ("fingerprint_auc@cifar10", [])
    assert resolve_metric_id("Detection-Rate", IDS) == ("detection_rate@cifar10", [])


def test_resolve_ambiguous_lists_candidates():
    mid, cands = resolve_metric_id("fingerprint_auc", IDS)
    assert mid is None and set(cands) == {"fingerprint_auc@cifar10", "fingerprint_auc@fmnist"}


# ---------------------------------------------------------------- end to end on the example record

@pytest.fixture
def store(tmp_path):
    path = tmp_path / "record.json"
    shutil.copy(EXAMPLE, path)
    s = JsonRecordStore(path)
    set_record_store(s)
    yield s
    set_record_store(None)


def test_example_record_and_write_back(store):
    out = run_validate_metric("fingerprint_auc@cifar10")
    assert out["verdict"] == "reproduced"
    saved = json.loads(store.path.read_text())["metrics"][0]["validation"]
    assert saved["verdict"] == "reproduced"


def test_example_unrun_metric(store):
    assert run_validate_metric("detection_rate")["verdict"] == "not_run"


def test_example_ambiguous_name(store):
    out = run_validate_metric("fingerprint_auc")
    assert out["verdict"] == "not_found" and len(out["candidates"]) == 3


def test_missing_record_does_not_raise(monkeypatch):
    set_record_store(None)
    monkeypatch.delenv("LANDSEER_RECORD_PATH", raising=False)
    assert run_validate_metric("x")["verdict"] == "not_found"


def test_langchain_tool_invocation(store):
    assert validate_metric.name == "validate_metric"
    out = validate_metric.invoke({"metric": "fingerprint_auc@cifar10"})
    assert out["verdict"] == "reproduced"
