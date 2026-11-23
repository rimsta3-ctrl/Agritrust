"""Statistical baseline and classical machine-learning detectors (scikit-learn, XGBoost)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .base import BaseDetector, subsample


class ZScoreBaseline(BaseDetector):
    """Rule-based baseline: largest |z-score| among the variables, over time AND across sensors.

    score = max_v max(|v_z|, |v_xdev| / robust_std(v_xdev on train)). No learning except the scale.
    Every model has to beat this one.
    """
    name, kind, trained_on = "zscore_baseline", "rule", "none (scale only)"

    def fit(self, X, y=None, groups=None):
        self.z_cols = [c for c in X.columns if c.endswith("_z")]
        self.x_cols = [c for c in X.columns if c.endswith("_xdev")]
        self.x_scale = {}
        for c in self.x_cols:
            mad = np.median(np.abs(X[c] - np.median(X[c])))
            self.x_scale[c] = 1.4826 * mad if mad > 0 else float(X[c].std() or 1.0)
        return self

    def score(self, X, groups=None):
        parts = [X[c].abs().to_numpy() for c in self.z_cols]
        parts += [(X[c].abs() / self.x_scale[c]).to_numpy() for c in self.x_cols]
        return np.max(np.vstack(parts), axis=0)


class IsolationForestDetector(BaseDetector):
    name, kind = "isolation_forest", "unsupervised"

    def fit(self, X, y=None, groups=None):
        Xn = self._check_columns(X)
        self.scaler = StandardScaler().fit(Xn)
        self.model = IsolationForest(n_estimators=self.params.get("n_estimators", 200),
                                     max_samples=self.params.get("max_samples", 512),
                                     random_state=self.seed, n_jobs=-1)
        self.model.fit(self.scaler.transform(Xn))
        return self

    def score(self, X, groups=None):
        return -self.model.score_samples(self.scaler.transform(self._check_columns(X)))


class OneClassSVMDetector(BaseDetector):
    """RBF One-Class SVM. Training cost grows ~quadratically, so it is fitted on a subsample."""
    name, kind = "one_class_svm", "unsupervised"

    def fit(self, X, y=None, groups=None):
        Xn = self._check_columns(X)
        Xs, _ = subsample(Xn, None, self.params.get("max_train_samples", 8000), self.seed)
        self.scaler = StandardScaler().fit(Xs)
        self.model = OneClassSVM(kernel="rbf", nu=self.params.get("nu", 0.02),
                                 gamma=self.params.get("gamma", "scale"))
        self.model.fit(np.clip(self.scaler.transform(Xs), -10, 10))
        return self

    def score(self, X, groups=None):
        Z = np.clip(self.scaler.transform(self._check_columns(X)), -10, 10)
        out = np.empty(len(Z))
        for i in range(0, len(Z), 50000):        # batches keep memory low
            out[i:i + 50000] = -self.model.decision_function(Z[i:i + 50000])
        return out


class RandomForestDetector(BaseDetector):
    """Supervised binary classifier trained on simulated attacks (upper bound / comparison)."""
    name, kind, trained_on = "random_forest", "supervised", "labelled simulated attacks"

    def fit(self, X, y=None, groups=None):
        Xn = self._check_columns(X)
        Xs, ys = subsample(Xn, np.asarray(y), self.params.get("max_train_samples"), self.seed)
        self.model = RandomForestClassifier(n_estimators=self.params.get("n_estimators", 200),
                                            min_samples_leaf=self.params.get("min_samples_leaf", 2),
                                            max_samples=self.params.get("max_samples"),
                                            class_weight="balanced_subsample", n_jobs=-1,
                                            random_state=self.seed)
        self.model.fit(Xs, ys)
        return self

    def score(self, X, groups=None):
        return self.model.predict_proba(self._check_columns(X))[:, 1]


class XGBoostDetector(BaseDetector):
    name, kind, trained_on = "xgboost", "supervised", "labelled simulated attacks"

    def fit(self, X, y=None, groups=None):
        from xgboost import XGBClassifier

        Xn = self._check_columns(X)
        Xs, ys = subsample(Xn, np.asarray(y), self.params.get("max_train_samples"), self.seed)
        pos = max(1, int(ys.sum()))
        self.model = XGBClassifier(n_estimators=self.params.get("n_estimators", 400),
                                   max_depth=self.params.get("max_depth", 6),
                                   learning_rate=self.params.get("learning_rate", 0.05),
                                   subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                   scale_pos_weight=(len(ys) - pos) / pos, eval_metric="aucpr",
                                   random_state=self.seed, n_jobs=-1)
        self.model.fit(Xs, ys)
        return self

    def score(self, X, groups=None):
        return self.model.predict_proba(self._check_columns(X))[:, 1]


# ----------------------------------------------------------------------------- stage 2
class TypeClassifier:
    """Stage 2: given a suspect row, say if it is an outlier, a sensor fault or a falsification."""

    def __init__(self, algo: str = "rf", seed: int = 0, **params):
        self.algo, self.seed, self.params = algo, seed, params
        self.name = f"{algo}_classifier"
        self.classes_: list[str] = []
        self.feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, categories: np.ndarray):
        self.feature_names = list(X.columns)
        self.classes_ = sorted(pd.unique(categories))
        y = np.array([self.classes_.index(c) for c in categories])
        Xn = X.to_numpy(dtype=np.float32)
        if self.algo == "rf":
            self.model = RandomForestClassifier(n_estimators=self.params.get("n_estimators", 300),
                                                min_samples_leaf=self.params.get("min_samples_leaf", 1),
                                                class_weight="balanced", n_jobs=-1, random_state=self.seed)
            self.model.fit(Xn, y)
        else:
            from sklearn.utils.class_weight import compute_sample_weight
            from xgboost import XGBClassifier

            self.model = XGBClassifier(n_estimators=self.params.get("n_estimators", 400),
                                       max_depth=self.params.get("max_depth", 6),
                                       learning_rate=self.params.get("learning_rate", 0.05),
                                       subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                       objective="multi:softprob", random_state=self.seed, n_jobs=-1)
            self.model.fit(Xn, y, sample_weight=compute_sample_weight("balanced", y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names].to_numpy(dtype=np.float32))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(self.classes_, dtype=object)[self.predict_proba(X).argmax(axis=1)]
