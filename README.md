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

## Reference

This project is motivated by the following paper:

> J Vahrenhold, J Niemann, K Drossel (2026). *Computational Thinking in ICILS 2023: Analyzing the Construct and Its Antecedent-and Process-Level Predictors*. ACM Transactions on Computing. https://dl.acm.org/doi/abs/10.1145/3813115

Paper author: Prof. Dr. Jan Vahrenhold, University of Münster (jan.vahrenhold@uni-muenster.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
