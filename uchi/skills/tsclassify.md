---
name: tsclassify
description: Classify fixed-length time series windows (ECG, HAR, sensor events)
args: <path.csv> [--label <col>]
mode: tsclassify
---

Each row of the CSV is one window's worth of time-series values with a label
column at the end (or specified via `--label`).

Trains a **TimeSeriesClassifier** and reports hold-out accuracy.

**Example**
```
/tsclassify ecg_windows.csv --label diagnosis
/tsclassify har_data.csv
/tsclassify fault_windows.csv --label fault_type
```

Natural language triggers:
> "classify these ECG windows"
> "what type of activity is this sensor window?"
