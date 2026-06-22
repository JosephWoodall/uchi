# Tabular ML and Preprocessing

## Preprocessing

**`FeatureDiscretizer`**

Converts any feature matrix to token sequences. Continuous features → equal-frequency quantile bins (tokens are bin indices). Categorical features → ordinal integers. Missing values and `NaN` → a special `__MISSING__` token. The result is a list of `(feature_index, bin)` tuples per row, which the trie can match exactly.

**`LabelEncoder`**

Bidirectional label ↔ integer mapping with `partial_fit` for new classes arriving at runtime. Used internally by all supervised classes.

---

## Tabular ML

**`TabularPredictor`** — classification

Encodes each row as a sequence of feature tokens, with the class label as the final token. The trie learns `P(label | feature_sequence)`. Three feature orderings are ensembled (MI-ascending, MI-descending, natural) to reduce ordering sensitivity. Prediction averages label distributions across all orderings.

sklearn-compatible: works in `Pipeline`, `GridSearchCV`, `cross_val_score`. Supports `partial_fit` for streaming or incremental learning.

```python
clf = TabularPredictor(n_bins=10, n_orderings=3)
clf.fit(X_train, y_train)
clf.predict(X_test)            # class labels
clf.predict_proba(X_test)      # list of {label: prob} dicts
clf.partial_fit(X_new, y_new)  # online update
```

**`TabularRegressor`** — regression

Same architecture as `TabularPredictor` but the continuous target is discretized into quantile bins. Prediction returns the credibility-weighted mean of bin centers. `predict_interval()` also returns the standard deviation of the bin distribution as a calibrated uncertainty estimate.

```python
reg = TabularRegressor(n_bins=10, n_target_bins=20)
reg.fit(X_train, y_train)
reg.predict(X_test)            # float means
reg.predict_interval(X_test)   # list of (mean, std) tuples
reg.score(X_test, y_test)      # R²
```
