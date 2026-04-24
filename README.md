# PaySim Fraud Detection Case Study

## Objective
Develop a transaction-level suspicious-activity scoring workflow for the PaySim dataset under severe class imbalance, while keeping false positives operationally manageable.

## Approach
The workflow was designed around three priorities: **leakage control**, **time-aware evaluation**, and **operational decisioning**.

The primary supervised label was `isFraud`. To reduce leakage risk, the main models excluded the four balance variables (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`). Raw identifiers (`nameOrig`, `nameDest`) were also excluded as direct inputs to avoid memorization. Instead, they were used only to derive behavioral features such as first-seen indicators, prior transaction-count features, and recent activity summaries.

The final feature family combined transaction type, log-transformed transaction amount, timeline position (`step`), hour-of-day, sender / destination novelty, and entity-history activity.

### Data split

| Split | Purpose |
|---|---|
| Train | Model fitting and tuning |
| Validation | Model selection and threshold selection |
| Holdout | Final one-time reporting |

A time-based split was used throughout so that model development remained forward-looking.

## Models

### Baseline: Logistic Regression
The baseline used a compact interpretable feature set: `step`, `type`, and `log_amount`. Additional feature blocks were then tested in a structured way. The best logistic specification added `hour_of_day`, `orig_is_first_seen`, `dest_is_first_seen`, `log_orig_prior_tx_count`, and `log_dest_prior_tx_count`. Its threshold was selected on validation and then locked before holdout testing.

### Advanced model: XGBoost
The advanced model used the same core feature family but allowed non-linear interactions among transaction type, amount, novelty, prior activity, and time structure.

Tuning followed a staged process:
1. random search with rolling time-based CV inside the training set,
2. stability filtering using fold variability and train-validation gap,
3. narrowed grid search around the strongest stable candidates,
4. validation-based comparison of the top candidates,
5. validation-based threshold selection.

### Final XGBoost configuration

| Hyperparameter | Value |
|---|---:|
| `max_depth` | 4 |
| `learning_rate` | 0.05 |
| `n_estimators` | 400 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.7 |
| `colsample_bylevel` | 0.8 |
| `gamma` | 0.5 |
| Threshold | 0.93 |

## Metrics and trade-offs
Because fraud is rare, **PR AUC** was used as the primary ranking metric and **ROC AUC** as a secondary metric.

Thresholds were not chosen using a default 0.5 cutoff. Instead, selection was based on operational metrics including expected cost, alert volume, false omission rate, and incremental alerts against the negative class.

### Cost function

| Error type | Cost |
|---|---:|
| False Negative (FN) | 500 |
| False Positive (FP) | 5 |

This reflected the business trade-off that missing fraud is much more costly than reviewing extra alerts.

## Final holdout results

| Metric | Best Logistic Regression | Final XGBoost |
|---|---:|---:|
| ROC AUC | 0.9510 | 0.9726 |
| PR AUC | 0.4532 | 0.6406 |
| Precision | 0.0508 | 0.2608 |
| Recall | 0.9256 | 0.8339 |
| F2 | 0.2082 | 0.5792 |
| Alert volume | 33,803 | 5,929 |
| Expected cost | 229,435 | 175,915 |

Overall, XGBoost materially outperformed the logistic benchmark. It achieved stronger ranking quality, much higher precision, far lower alert volume, and lower expected cost on the same holdout period.

## Interpretation and practical findings
SHAP analysis showed that the final XGBoost model was driven primarily by transaction type, transaction amount, destination novelty, destination-side prior history, and temporal structure (`step` and `hour_of_day`).

The top 20 highest-risk holdout transactions were all true frauds, indicating strong ranking quality at the top of the alert stack. These cases were dominated by large `CASH_OUT` transactions with sparse prior history and frequent first-seen destination patterns.

Segment analysis showed the model was strongest in the most operationally relevant groups:

| Segment view | Main finding |
|---|---|
| Transaction type | Strongest performance in `TRANSFER` / `CASH_OUT` |
| Destination novelty | Better performance when `dest_is_first_seen = 1` |
| Amount band | Strongest performance in the highest transaction-amount band |

## Conclusion
The final workflow combined leakage-aware feature engineering, time-based validation, interpretable benchmarking, stability-aware XGBoost tuning, and business-oriented threshold selection. The final XGBoost model provided the strongest overall balance of ranking performance and operational usability on the holdout set.
