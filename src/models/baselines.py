"""Baselines on MFCC summary statistics. All share fit(X, y) / predict_proba(X)."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_baseline(name, seed, params=None):
    params = params or {}
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=params.get("C", 1.0), class_weight="balanced", max_iter=5000))
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            class_weight="balanced", n_jobs=-1, random_state=seed)
    raise ValueError(f"unknown baseline {name}")


# Models whose result does not depend on the seed: run them once, not once per seed.
DETERMINISTIC = {"majority", "logreg"}
