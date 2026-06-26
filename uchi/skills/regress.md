---
name: regress
description: Predict a numeric target column from tabular features
args: <path.csv> [--target <col>]
mode: regress
---

Trains a **TabularRegressor** (trie-based regression) and reports mean absolute
error on the hold-out set. Specify the target column with `--target`; defaults
to the last column or a column named `target`.

**Example**
```
/regress housing.csv --target price
/regress sales.csv --target revenue
/regress energy.csv --target consumption
```

Natural language triggers:
> "predict the price in housing.csv"
> "regression on my sales data"
