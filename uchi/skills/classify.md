---
name: classify
description: Classify tabular data rows using the trie predictor
args: <path.csv> [--label <col>]
mode: classify
---

Trains a **TabularPredictor** (trie-based classifier) on your data and reports
hold-out accuracy. The last column is used as the label unless you specify
`--label <col>`.

**Example**
```
/classify iris.csv --label species
/classify customers.csv --label churned
/classify titanic.csv --label survived
```

You can also describe what you want in natural language:
> "classify my customer data and predict churn"
> "predict which rows belong to which category in data.csv"
