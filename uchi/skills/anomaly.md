---
name: anomaly
description: Detect anomalous rows in tabular or time series data
args: <path.csv>
mode: anomaly
---

Trains an **AnomalyDetector** on your data and flags rows whose prediction
surprise score (−log₂ P) exceeds mean + 2σ.

Returns row indices, count, and the top-scoring anomalous rows.

**Example**
```
/anomaly sensor_readings.csv
/anomaly network_traffic.csv
/anomaly transactions.csv
```

Natural language triggers:
> "detect anomalies in sensor_readings.csv"
> "are there any outliers in my data?"
> "find unusual patterns in transactions.csv"
