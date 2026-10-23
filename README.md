# Tuned gradient boosting vs. linear regression for student achievement

## Motivation
The paper "Computational Thinking in ICILS 2023: Analyzing the Construct and Its Antecedent-and Process-Level Predictors" studies antecedent (background) and process-level predictors of achievement. Such findings are typically linear. This experiment is a first, small test of whether they hold up under a nonlinear, cross-validation-tuned model. It uses a public stand-in dataset, since ICILS 2023 is not used here.

## Question
Can a cross-validation-tuned gradient-boosting model predict student achievement from background and study-process predictors more accurately than a standard linear regression?

## Dataset
UCI Student Performance (id 320), fetched with `ucimlrepo`. The target is the final grade G3. The prior grades G1 and G2 are dropped. Predictors include parental education, family support, study time, past failures, absences and going out. This is **not** the paper's own dataset.

## Baseline vs. improved
- **Baseline:** ordinary linear regression on one-hot encoded, standardized predictors. It is a generic stand-in, not the paper's method.
- **Improved:** LightGBM on the same features and split, tuned by randomized 5-fold CV search on the training part only. It is run with seeds 42, 43 and 44.
- One fixed 75/25 split. The test set is used once, at the end.
- Main metric: test RMSE. Secondary metrics: MAE and R².

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On the UCI Student Performance dataset (649 rows, one 75/25 split), a standard linear regression reached test RMSE 2.68 grade points, while a CV-tuned LightGBM reached 2.66, a small gain of 0.02 that may be within noise.

| Metric | Value |
|---|---|
| baseline_score | 2.679 |
| improved_score | 2.657 |
| improvement | 0.02273 |
| baseline_score_std | 0 |
| improved_score_std | 0.02061 |
| baseline_mae | 2.087 |
| improved_mae | 2.079 |
| baseline_r2 | 0.1552 |
| improved_r2 | 0.1694 |
| improved_mae_std | 0.02271 |
| improved_r2_std | 0.01289 |
| improved_cv_rmse_mean | 2.161 |
| target_std_test | 2.924 |

Run on 2026-10-23. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in dataset, not ICILS 2023; the baseline is a generic linear model, not the paper's method. Single small split (test set of modest size) so differences may be within noise; seed std for LightGBM reflects model/CV randomness only, not split variability. OLS is deterministic (std 0).

## Reference

This project is motivated by the following paper:

> J Vahrenhold, J Niemann, K Drossel (2026). *Computational Thinking in ICILS 2023: Analyzing the Construct and Its Antecedent-and Process-Level Predictors*. ACM Transactions on Computing. https://dl.acm.org/doi/abs/10.1145/3813115

Paper author: Prof. Dr. Jan Vahrenhold, University of Münster (jan.vahrenhold@uni-muenster.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
