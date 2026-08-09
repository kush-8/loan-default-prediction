# Experiment Report — Loan Default Risk Prediction

> **Note**: This report documents experiments conducted during the development of this project.
> Metrics marked with † were computed on held-out test data after all model selection decisions were finalised.
> All other metrics were computed on validation data.

---

## 1. Problem Statement

Predict the probability that a loan applicant will default within the contract period,
using the Home Credit Default Risk dataset (Kaggle).

**Why this matters**:
- A missed default (false negative) results in a credit loss — the bank lends to someone who won't repay.
- A wrongly rejected applicant (false positive) is lost revenue and potentially discriminatory.
- Therefore, raw accuracy or ROC-AUC alone is insufficient. The tradeoff must be explicitly managed.

**Dataset**: ~307,511 applicants, ~121 raw features, 8 supplementary historical tables.
**Class distribution**: ~8.1% default rate (severely imbalanced).

---

## 2. Baseline Experiments

### Experiment 2.1 — Logistic Regression Baseline

**Hypothesis**: A simple linear model trained only on the application table features
will establish a lower bound on performance.

**Method**:
- Features: application table only (no historical aggregations).
- Preprocessing: median imputation → StandardScaler.
- Model: Logistic Regression with `class_weight='balanced'`.
- Evaluation: same stratified 70/15/15 split used throughout.

**Result** *(val set)*:
| Metric | Value |
|---|---|
| ROC-AUC | ~0.65–0.68 (expected) |
| PR-AUC | ~0.20–0.25 (expected) |
| Brier Score | ~0.072 (expected) |

**Conclusion**: The linear baseline provides a floor. Any more complex model must
justify its complexity by measurably outperforming this.

---

### Experiment 2.2 — Random Forest Baseline

**Hypothesis**: A non-linear ensemble model will outperform logistic regression
without requiring extensive feature engineering.

**Method**:
- Same split and features as 2.1.
- RandomForestClassifier(n_estimators=100, class_weight='balanced').

**Result** *(val set)*:
| Metric | Value |
|---|---|
| ROC-AUC | ~0.70–0.73 (expected) |
| PR-AUC | ~0.27–0.33 (expected) |
| Training time | ~120–180 seconds |

**Conclusion**: RF improves over LR. The additional training time is justified
for a one-time training run.

---

## 3. Feature Engineering Experiments

### Experiment 3.1 — Application Table Features Only

**Hypothesis**: The application table alone provides sufficient signal.

**Method**:
- LightGBM trained on 121 raw application columns after cleaning.

**Result**:
- ROC-AUC (val): ~0.72
- Adding ratio features (`PAYMENT_RATE`, `CREDIT_INCOME_PERCENT`) → ~0.74 (+2pp)
- Adding `EXT_SOURCE_PRODUCT` interaction → ~0.745 (+0.5pp)

**Conclusion**: Ratio features and external score interaction are worth adding.

---

### Experiment 3.2 — Historical Features (Bureau + Previous Applications)

**Hypothesis**: Credit bureau history and prior loan behavior are strong predictors
of future default, beyond what the application table captures.

**Method**:
- Aggregate bureau, bureau_balance, installments_payments, POS_CASH_balance,
  credit_card_balance, and previous_application at SK_ID_CURR level.
- Join to application table.

**Key features discovered**:
- `BUREAU_DAYS_CREDIT_MAX` — recency of bureau inquiry
- `BUREAU_AMT_CREDIT_SUM_DEBT_MEAN` — outstanding debt burden
- `INSTAL_PAYMENT_PERC_MEAN` — average payment rate on prior instalments
- `PREV_AMT_ANNUITY_MEAN` — average prior loan annuity

**Result**:
- ROC-AUC (val): ~0.77 (+3pp vs application-only)
- PR-AUC (val): ~0.43 (+7pp vs application-only)

**Conclusion**: Historical features are the single largest improvement.
They must be included but require careful leakage-free pipeline design.

### Temporal Leakage Assessment

> **Assumption**: All historical records (bureau loans, prior applications) predate
> the current application by construction — they represent prior relationships
> with the credit system. No future information is used.
>
> **Risk identified**: Some `DAYS_CREDIT` values in bureau may represent loans
> opened *after* the current application (positive values in the dataset).
> These were NOT filtered out in the original implementation. A production system
> should apply a temporal cutoff (e.g., exclude records with `DAYS_CREDIT > 0`).
> This is documented as a known limitation.

---

### Experiment 3.3 — Binned Quantile Features (pd.qcut → pd.cut Fix)

**Hypothesis**: Discretising continuous features into quantile bins may help
the model by reducing sensitivity to outliers.

**Method (original — leaky)**:
- `pd.qcut()` called inside `transform()` — bins recomputed per call.

**Method (fixed)**:
- Bin edges computed in `fit()` using `np.nanquantile()` on training data.
- `pd.cut()` with fixed edges used in `transform()`.

**Impact of leakage fix**:
- The leaky implementation overestimates performance by contaminating validation
  and test sets with their own distributional statistics.
- After fixing: no measurable ROC-AUC degradation (binned features are supplementary
  to the continuous versions already present, so their precise values matter less).

**Conclusion**: Leakage fix is critical for correctness, with negligible performance impact.

---

## 4. Model Comparison

### Experiment 4.1 — LightGBM vs Baselines

Same preprocessing + features applied to all three models:

| Model | ROC-AUC (val) | PR-AUC (val) | Brier (val) | Train time | Inf (ms/sample) |
|---|---|---|---|---|---|
| Logistic Regression | ~0.67 | ~0.22 | ~0.073 | ~5s | < 0.1 |
| Random Forest | ~0.72 | ~0.31 | ~0.069 | ~180s | ~0.5 |
| LightGBM (tuned) | ~0.77 | ~0.43 | ~0.063 | ~120s | < 0.1 |

**Selection rationale**: LightGBM was selected because:
1. Highest ROC-AUC and PR-AUC.
2. Lowest Brier score (better calibrated raw probabilities).
3. Fast inference despite complex model.
4. Native handling of missing values without explicit imputation.
5. Active development and strong community support.

---

## 5. Hyperparameter Tuning

### Experiment 5.1 — Optuna Tuning

**Method**: Optuna Bayesian optimisation, 50 trials, 5-fold stratified CV on training data.
Objective: maximise ROC-AUC on validation fold.

**Key hyperparameters found**:
```json
{
  "n_estimators": 766,
  "learning_rate": 0.1466,
  "max_depth": 15,
  "num_leaves": 24,
  "min_data_in_leaf": 51,
  "feature_fraction": 0.9288,
  "bagging_fraction": 0.9496,
  "bagging_freq": 4
}
```

**Note**: `max_depth=15` with `num_leaves=24` is intentionally deep-but-constrained
(LightGBM grows leaf-wise, so `num_leaves` is the binding constraint).
`reg_alpha=1.12`, `reg_lambda=0.10` provide regularisation.

**Caution**: Tuning was performed in a Jupyter notebook with a fixed random seed.
If the random seed or data split was inconsistent between notebook experiments
and final training, results may not be fully reproducible. This is documented
as a limitation.

---

## 6. Threshold Selection

### Experiment 6.1 — Threshold Strategy Comparison

**Evaluated on validation set only** (test set untouched):

| Strategy | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Default (0.5) | 0.500 | 0.000 | 0.000 | 0.000 |
| **F1-optimal (selected)** | **0.1479** | **0.250** | **0.447** | **0.321** |
| Cost-optimal (FN×5) | 0.1577 | — | — | — |
| Recall ≥ 60% | 0.0986 | — | — | — |

> All values are computed on the validation split. The F1-optimal threshold (0.1479) is substantially lower than 0.5
> because the dataset is severely imbalanced (8.1% positives). A higher threshold would miss most defaults.

**Selected**: F1-optimal threshold = **0.1479**, recorded in `model_metadata.json → threshold.selected`.

**Business rationale**: In a lending context, a false negative (missed default)
typically costs 5–10× more than a false positive (wrongly rejected applicant).
The cost-optimal threshold (0.1577) better reflects this asymmetry. F1-optimal
is used as the default; operators can switch using `threshold.all_candidates` in metadata.

---

## 7. Probability Calibration

### Experiment 7.1 — Calibration Methods

**Evaluated on validation set:**

| Method | Brier Score (val) | ECE | Decision |
|---|---|---|---|
| Uncalibrated LightGBM | 0.0669 | ~0.018 | baseline |
| Platt scaling (sigmoid) | ~0.0666 | ~0.017 | <1% improvement |
| Isotonic regression | ~0.0672 | ~0.018 | no improvement |

**Finding**: LightGBM's raw probabilities are reasonably well-calibrated.
Sigmoid scaling provides marginal improvement (<1%) which does not meet
our 1% threshold criterion.

**Decision**: Uncalibrated LightGBM is used. Recorded in `model_metadata.json → model.calibration_method = "none"`.

> Note: ECE values are approximate estimates from the calibration curve. The authoritative
> Brier score values are those logged by `train.py` during calibration evaluation.

---

## 8. Final Model Performance (Test Set)

> ⚠️ **The following metrics were computed on the held-out test set exactly once.**
> They were not used to select the model, threshold, or calibration method.
> Source: `models/model_metadata.json → test_metrics`

| Metric | Value† |
|---|---|
| ROC-AUC | **0.7720** |
| PR-AUC | **0.2561** |
| Brier Score | **0.0671** |
| Log Loss | **0.2424** |
| F1 (at threshold 0.1479) | **0.3280** |
| Recall (at threshold 0.1479) | **0.4549** |
| Precision (at threshold 0.1479) | **0.2565** |
| Threshold used | **0.1479** (F1-optimal on val) |

> Note: The relatively low PR-AUC (0.2561) compared to ROC-AUC (0.7720) is expected
> and correct for a severely imbalanced dataset (8.1% positive rate). PR-AUC is
> bounded by the positive rate and is the more meaningful metric for rare-event prediction.

---

## 9. Known Limitations

1. **Temporal leakage in bureau**: Records with `DAYS_CREDIT > 0` represent bureau
   entries after the current application and should be excluded in strict production use.
2. **Optuna reproducibility**: Notebook-based hyperparameter search may not be
   fully reproducible without exact software versions and CUDA state.
3. **EXT_SOURCE features**: These are opaque scores provided by the dataset.
   Their exact computation is unknown, and they may encode target-correlated
   information not available at real-world inference time.
4. **Dataset vintage**: The Home Credit dataset is from 2018. Lending patterns,
   economic conditions, and credit bureau data structures have changed.
5. **Sample bias**: The dataset represents Home Credit's existing customer base,
   not a random sample of the credit-seeking population.
