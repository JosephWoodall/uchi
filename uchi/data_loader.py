"""
data_loader.py
==============
Lightweight data-loading utilities for Uchi's analytical skill handlers.
Supports CSV and JSON with no dependencies beyond stdlib.
"""
from __future__ import annotations

import csv
import json
import os
import re
from typing import List, Optional, Tuple

# Matches a data file path in free-form user text
_PATH_RE = re.compile(
    r'["\']?(/?\S+\.(csv|json|tsv|npy|parquet|feather|xlsx))["\']?',
    re.IGNORECASE,
)


def find_path(text: str) -> Optional[str]:
    """Extract the first data-file path from a free-form string."""
    m = _PATH_RE.search(text)
    return m.group(1) if m else None


def parse_args(args: str) -> dict:
    """
    Parse 'path.csv [--label col] [--target col] [--steps N]'.
    Returns dict with keys: path, label, target, steps.
    """
    result: dict = {"path": "", "label": None, "target": None, "steps": 10}
    parts = args.split()
    i = 0
    if parts and not parts[0].startswith("--"):
        result["path"] = parts[0]
        i = 1
    while i < len(parts):
        flag = parts[i]
        if flag in ("--label", "--target", "--steps") and i + 1 < len(parts):
            val = parts[i + 1]
            key = flag[2:]
            result[key] = int(val) if key == "steps" else val
            i += 2
        else:
            i += 1
    try:
        result["steps"] = int(result["steps"])
    except (TypeError, ValueError):
        result["steps"] = 10
    return result


def load_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    """Load CSV → (header, raw string rows). Auto-detects header row."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        return [], []
    # First row is header if it contains any non-numeric value
    try:
        [float(v) for v in rows[0] if v.strip()]
        header = [f"col_{i}" for i in range(len(rows[0]))]
        data_rows = rows
    except ValueError:
        header = [h.strip() for h in rows[0]]
        data_rows = rows[1:]
    return header, data_rows


def load_data(path: str) -> Tuple[List[str], List[List[str]]]:
    """Load CSV or JSON → (header, raw string rows)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        return load_csv(path)
    elif ext == ".json":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            header = list(obj[0].keys())
            rows = [[str(row.get(k, "")) for k in header] for row in obj]
            return header, rows
        raise ValueError("JSON must be a list of dicts")
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .csv or .json")


def _label_col_index(header: List[str], label_col: Optional[str] = None) -> int:
    """Return index of the label/target column."""
    if label_col:
        lo = label_col.lower()
        for i, h in enumerate(header):
            if h.lower() == lo:
                return i
        raise ValueError(f"Column '{label_col}' not found. Available: {header}")
    # Heuristic: look for well-known label column names
    for i, h in enumerate(header):
        if h.lower() in {"label", "target", "y", "class", "output", "result"}:
            return i
    return len(header) - 1  # default: last column


def split_features(
    header: List[str],
    rows: List[List[str]],
    label_col: Optional[str] = None,
) -> Tuple[List[List[float]], List]:
    """
    Split rows into X (numeric 2D list) and y (raw labels, possibly strings).
    Rows where feature values are non-numeric are skipped silently.
    """
    if not rows:
        raise ValueError("No data rows")
    idx = _label_col_index(header, label_col)
    X, y = [], []
    for row in rows:
        if len(row) < len(header):
            continue
        y_val = row[idx].strip() if isinstance(row[idx], str) else str(row[idx])
        x_parts: List[float] = []
        ok = True
        for i, v in enumerate(row[: len(header)]):
            if i == idx:
                continue
            try:
                x_parts.append(float(v))
            except (ValueError, TypeError):
                ok = False
                break
        if ok and x_parts:
            X.append(x_parts)
            y.append(y_val)
    return X, y


def to_numeric_rows(
    header: List[str], rows: List[List[str]]
) -> List[List[float]]:
    """Convert all columns to float, skipping rows with non-numeric values."""
    result = []
    for row in rows:
        try:
            result.append([float(v) for v in row[: len(header)]])
        except (ValueError, TypeError):
            pass
    return result


def train_test_split(
    X: list,
    y: list,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[list, list, list, list]:
    """Deterministic train/test split. Returns X_train, X_test, y_train, y_test."""
    import random

    rng = random.Random(seed)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    cut = int(len(indices) * (1 - test_frac))
    tr, te = indices[:cut], indices[cut:]
    return (
        [X[i] for i in tr],
        [X[i] for i in te],
        [y[i] for i in tr],
        [y[i] for i in te],
    )
