---
name: forecast
description: Auto-regressive multi-step forecast for time series data
args: <path.csv> [--steps N]
mode: forecast
---

Fits a **MultivariateTSPredictor** (trie-based step-ahead predictor) on a
CSV of time-ordered rows and forecasts the next N steps (default 10).

Each row of the CSV is one timestep; columns are dimensions.

**Example**
```
/forecast prices.csv --steps 20
/forecast sensor_log.csv
/forecast energy_hourly.csv --steps 48
```

Natural language triggers:
> "forecast the next 20 steps in prices.csv"
> "what will my sensor read next week?"
> "predict future values for energy_hourly.csv"
