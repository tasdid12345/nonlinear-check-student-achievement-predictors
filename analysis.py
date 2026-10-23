"""Student achievement (G3): tuned LightGBM vs. standard linear regression (UCI Student Performance, id 320)."""
import json, os, sys, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

SEEDS = [42, 43, 44]
QUESTION = ("Can a cross-validation-tuned gradient-boosting model predict student achievement from background and "
            "study-process predictors more accurately than a standard linear regression?")
METHOD = ("UCI Student Performance (id 320); target G3, G1/G2 dropped; one fixed 75/25 split; tuning by 5-fold CV on "
          "train only; baseline one-hot+standardized OLS vs. randomized-search-tuned LightGBM; metric test RMSE.")


def load():
    """Download with retry/backoff (ucimlrepo uses requests internally)."""
    for attempt in range(5):
        try:
            return fetch_ucirepo(id=320)
        except Exception as e:  # network / HTTP 429 / 5xx
            wait = 2 ** attempt * 3
            print(f"download failed ({e!r}); retry in {wait}s", flush=True)
            time.sleep(wait)
    sys.exit("could not download dataset")


def make_pre(X):
    """Shared preprocessing: one-hot for categoricals, standardization for numerics."""
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = X.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                              ("num", StandardScaler(), num)])


def evaluate(model, Xte, yte):
    """Shared evaluation function: RMSE (main), MAE, R2."""
    p = model.predict(Xte)
    return {"rmse": float(np.sqrt(mean_squared_error(yte, p))),
            "mae": float(mean_absolute_error(yte, p)), "r2": float(r2_score(yte, p))}


ds = load()
X = ds.data.features.copy()
y = ds.data.targets.iloc[:, 0] if isinstance(ds.data.targets, pd.DataFrame) else ds.data.targets
X = X.drop(columns=[c for c in ["G1", "G2"] if c in X.columns])  # drop prior grades
ok = X.notna().all(axis=1) & y.notna()
X, y = X[ok].reset_index(drop=True), y[ok].reset_index(drop=True).astype(float)
print(f"data: {X.shape[0]} rows, {X.shape[1]} predictors", flush=True)
if len(X) < 200:
    sys.exit("too little data")

# One fixed split (seed 42); the test set is used only for the final evaluation.
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"train {len(Xtr)}, test {len(Xte)}", flush=True)

# Baseline: deterministic OLS
base = Pipeline([("pre", make_pre(X)), ("lr", LinearRegression())]).fit(Xtr, ytr)
b = evaluate(base, Xte, yte)
print("baseline", b, flush=True)

# Improved: LightGBM tuned by randomized 5-fold CV on train only, for each seed
grid = {"lgbm__n_estimators": [100, 200, 400], "lgbm__learning_rate": [0.01, 0.03, 0.1],
        "lgbm__num_leaves": [4, 8, 16], "lgbm__min_child_samples": [10, 20, 40],
        "lgbm__subsample": [0.7, 1.0], "lgbm__colsample_bytree": [0.6, 1.0]}
runs, cv_rmse = [], []
for s in SEEDS:
    pipe = Pipeline([("pre", make_pre(X)),
                     ("lgbm", LGBMRegressor(subsample_freq=1, random_state=s, n_jobs=1, verbose=-1))])
    search = RandomizedSearchCV(pipe, grid, n_iter=20, cv=KFold(5, shuffle=True, random_state=s),
                                scoring="neg_root_mean_squared_error", random_state=s, n_jobs=-1)
    search.fit(Xtr, ytr)
    m = evaluate(search.best_estimator_, Xte, yte)
    runs.append(m)
    cv_rmse.append(float(-search.best_score_))
    print(f"seed {s}: CV RMSE {cv_rmse[-1]:.3f}, test {m}, params {search.best_params_}", flush=True)

imp = {k: float(np.mean([r[k] for r in runs])) for k in b}
imp_std = {k: float(np.std([r[k] for r in runs])) for k in b}
metrics = {
    "baseline_score": b["rmse"], "improved_score": imp["rmse"],
    "improvement": b["rmse"] - imp["rmse"],
    "baseline_score_std": 0.0, "improved_score_std": imp_std["rmse"],
    "baseline_mae": b["mae"], "improved_mae": imp["mae"],
    "baseline_r2": b["r2"], "improved_r2": imp["r2"],
    "improved_mae_std": imp_std["mae"], "improved_r2_std": imp_std["r2"],
    "improved_cv_rmse_mean": float(np.mean(cv_rmse)),
    "target_std_test": float(yte.std()),
}
if not all(np.isfinite(v) for v in metrics.values()):
    sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
for a, k, t in zip(ax, ["rmse", "mae"], ["Test RMSE", "Test MAE"]):
    a.bar(["Linear reg.", "LightGBM"], [b[k], imp[k]], yerr=[0, imp_std[k]], color=["gray", "tab:green"])
    a.set_title(t + " (lower is better)")
plt.tight_layout()
plt.savefig("figures/baseline_vs_improved.png", dpi=120)

results = {
    "question": QUESTION, "method": METHOD,
    "dataset": {"name": "UCI Student Performance (id 320)",
                "source_url": "https://archive.ics.uci.edu/dataset/320/student+performance",
                "n_rows": int(len(X)), "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "Test RMSE (G3 grade points)", "higher_is_better": False,
    "baseline": "Ordinary linear regression on one-hot encoded + standardized predictors (all predictors, G1/G2 excluded).",
    "improved": "LightGBM regressor on the same preprocessed features, tuned by 20-iteration randomized 5-fold CV search on the training set; mean over seeds 42-44.",
    "metrics": metrics,
    "notes": ("Stand-in dataset, not ICILS 2023; the baseline is a generic linear model, not the paper's method. "
              "Single small split (test set of modest size) so differences may be within noise; "
              "seed std for LightGBM reflects model/CV randomness only, not split variability. "
              "OLS is deterministic (std 0)."),
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(json.dumps(metrics, indent=2))
