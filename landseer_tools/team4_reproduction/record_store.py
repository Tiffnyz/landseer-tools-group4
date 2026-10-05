"""Where Validate_metric reads the metrics record from.

Team 2 owns the record. Until their storage is final, this module provides a
JSON-file store with the same shape. To switch to Team 2's implementation,
write a class with the same three methods and pass it to `set_record_store()`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Protocol

from .schemas import MetricEntry, MetricsRecord, ValidationResult


class RecordStore(Protocol):
    def list_metric_ids(self) -> list[str]: ...
    def get_entry(self, metric_id: str) -> MetricEntry: ...
    def save_validation(self, metric_id: str, result: ValidationResult) -> None: ...


class JsonRecordStore:
    """Stores a MetricsRecord as one JSON file."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def _load(self) -> MetricsRecord:
        return MetricsRecord.model_validate_json(self.path.read_text())

    def list_metric_ids(self) -> list[str]:
        return [m.metric_id for m in self._load().metrics]

    def get_entry(self, metric_id: str) -> MetricEntry:
        for m in self._load().metrics:
            if m.metric_id == metric_id:
                return m
        raise KeyError(metric_id)

    def save_validation(self, metric_id: str, result: ValidationResult) -> None:
        record = self._load()
        for m in record.metrics:
            if m.metric_id == metric_id:
                m.validation = result.model_dump(mode="json")
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(record.model_dump_json(indent=2))
        tmp.replace(self.path)  # atomic, so a crash can't leave a half-written record


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9@.]", "", s.lower())


def resolve_metric_id(query: str, ids: list[str]) -> tuple[Optional[str], list[str]]:
    """Match what the agent typed to a metric id.

    Accepts the exact id, or a name that matches exactly one id
    ('fingerprint_auc' -> 'fingerprint_auc@cifar10' if it's the only one).
    Returns (id, []) on success, or (None, candidates) if not found or ambiguous.
    """
    if query in ids:
        return query, []
    q = _norm(query)
    exact = [i for i in ids if _norm(i) == q]
    if len(exact) == 1:
        return exact[0], []
    by_name = [i for i in ids if _norm(i.split("@")[0]) == q]
    if len(by_name) == 1:
        return by_name[0], []
    if by_name:
        return None, by_name
    partial = [i for i in ids if q and q in _norm(i)]
    return None, partial or ids


# ---------------------------------------------------------------- configured store

_store: Optional[RecordStore] = None


def set_record_store(store: RecordStore) -> None:
    global _store
    _store = store


def get_record_store() -> RecordStore:
    """The configured store, or a JSON store from $LANDSEER_RECORD_PATH."""
    if _store is not None:
        return _store
    path = os.environ.get("LANDSEER_RECORD_PATH")
    if not path:
        raise RuntimeError("No metrics record configured. Call set_record_store() "
                           "or set LANDSEER_RECORD_PATH to the record JSON file.")
    return JsonRecordStore(path)
