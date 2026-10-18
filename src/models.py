"""
Training, tuning and comparing nine classifiers on a prepared dataset.

Revised 2026-10-18:

- Optuna scored each trial by the model's accuracy on the rows it had just been trained on, which
  rewards memorising: 180 of the 380 trials in the notebook scored exactly 1.000, so the tuner
  could not tell those settings apart and kept whichever got there first. Each trial is now scored
  by 5-fold cross-validation on the training rows.
- Five of the nine models had no settings to search, so their 20 trials were 20 identical fits.
  Each now has a search space.
- SMOTE ran once over the whole training set, so a later cross-validation, like the notebook's,
  scored folds on synthetic rows made from their own neighbours. Preprocessing, SMOTE and the model
  are now one pipeline, fitted inside each fold, and SMOTE never touches a row it is scored on.
- The leaderboard ranked the models by their score on the test rows and reported the top one's,
  which is partly luck on 20% of the data: on ten synthetic datasets it was 3.0 points from the
  winner's true accuracy on average. Models are now ranked by cross-validation (0.8 points); the
  held-out rows are scored once, at the end, and decide nothing.
- ROC AUC took the second column of the probabilities, which raised an error with three or more
  classes and was reported as 0.0; the ROC curve then crashed the run. Multi-class AUC is the
  one-against-the-rest average, and each class gets its own curve.
- The outputs folder was assumed to exist, so a fresh clone crashed after training every model.
- SVM fitting time grows with at least the square of the rows; above 10,000 training rows it is
  skipped with a note instead of running for hours.
"""

import os
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

MODELS = ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "LogisticRegression", "KNN", "SVM",
          "ExtraTrees", "NaiveBayes"]
FOLDS = 5
SVM_MAX_ROWS = 10_000
SEED = 42


def make(name, p):
    """A fresh classifier with settings `p`, single-threaded (the folds run in parallel)."""
    if name == "RandomForest":
        return RandomForestClassifier(**p, class_weight="balanced", random_state=SEED, n_jobs=1)
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(**p, eval_metric="logloss", random_state=SEED, n_jobs=1)
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**p, random_state=SEED, n_jobs=1, verbose=-1)
    if name == "CatBoost":
        import catboost as cb
        return cb.CatBoostClassifier(**p, verbose=0, random_seed=SEED, thread_count=1)
    if name == "LogisticRegression":
        return LogisticRegression(**p, max_iter=2000, class_weight="balanced", random_state=SEED)
    if name == "KNN":
        return KNeighborsClassifier(**p)
    if name == "SVM":
        return SVC(**p, probability=True, random_state=SEED)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(**p, class_weight="balanced", random_state=SEED, n_jobs=1)
    if name == "NaiveBayes":
        return GaussianNB(**p)
    raise ValueError(f"Unknown model: {name}")


def space(trial, name):
    """The settings Optuna searches for each model; the first four as before."""
    if name == "RandomForest":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 300),
                    max_depth=trial.suggest_int("max_depth", 3, 20))
    if name in ("XGBoost", "LightGBM"):
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 300),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                    max_depth=trial.suggest_int("max_depth", 3, 10))
    if name == "CatBoost":
        return dict(iterations=trial.suggest_int("iterations", 100, 300),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                    depth=trial.suggest_int("depth", 3, 10))
    if name == "LogisticRegression":
        return dict(C=trial.suggest_float("C", 1e-3, 1e2, log=True))
    if name == "KNN":
        return dict(n_neighbors=trial.suggest_int("n_neighbors", 1, 40),
                    weights=trial.suggest_categorical("weights", ["uniform", "distance"]))
    if name == "SVM":
        return dict(C=trial.suggest_float("C", 1e-2, 1e2, log=True),
                    gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True))
    if name == "ExtraTrees":
        return dict(n_estimators=trial.suggest_int("n_estimators", 100, 300),
                    max_depth=trial.suggest_int("max_depth", 3, 20))
    if name == "NaiveBayes":
        return dict(var_smoothing=trial.suggest_float("var_smoothing", 1e-11, 1e-1, log=True))
    raise ValueError(f"Unknown model: {name}")


def smote_neighbours(y, folds=FOLDS):
    """SMOTE's neighbour count for these labels: 5, or fewer when a class is too small to have 5
    neighbours inside a fold. 0 means a class is too small for SMOTE at all."""
    smallest = np.bincount(y).min()
    in_fold = smallest - int(np.ceil(smallest / folds))
    return int(max(0, min(5, in_fold - 1)))


def pipeline(prepared, name, params, smote=True):
    steps = [("prep", prepared.preprocessor())]
    k = smote_neighbours(prepared.y_train)
    if smote and k > 0:
        steps.append(("smote", SMOTE(k_neighbors=k, random_state=SEED)))
    steps.append(("model", make(name, params)))
    return Pipeline(steps)


def folds_for(y):
    return StratifiedKFold(min(FOLDS, max(2, np.bincount(y).min())), shuffle=True, random_state=SEED)


def tune(prepared, name, trials=20, smote=True, n_jobs=-1):
    """Search `name`'s settings, scoring each trial by cross-validated accuracy on the training rows.

    Returns (best settings, mean accuracy over the folds, its standard deviation).
    """
    cv = folds_for(prepared.y_train)

    def objective(trial):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipeline(prepared, name, space(trial, name), smote),
                                     prepared.X_train, prepared.y_train, cv=cv, scoring="accuracy",
                                     n_jobs=n_jobs)
        trial.set_user_attr("sd", float(scores.std()))
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    return study.best_params, study.best_value, study.best_trial.user_attrs["sd"]


def auc(model, X, y, n_classes):
    """ROC AUC on these rows: binary as usual, several classes as the average of each class against
    the rest (over the classes present in `y`)."""
    proba = model.predict_proba(X)
    if n_classes == 2:
        return roc_auc_score(y, proba[:, 1])
    present = [c for c in range(n_classes) if 0 < np.sum(y == c) < len(y)]
    return float(np.mean([roc_auc_score(y == c, proba[:, c]) for c in present]))


def roc_curves(model, prepared):
    """[(label, false-positive rates, true-positive rates, AUC)]: one curve for two classes (the
    second class as positive), one per class otherwise."""
    proba = model.predict_proba(prepared.X_test)
    y, n = prepared.y_test, len(prepared.classes)
    targets = [1] if n == 2 else [c for c in range(n) if 0 < np.sum(y == c) < len(y)]
    out = []
    for c in targets:
        fpr, tpr, _ = roc_curve(y == c, proba[:, c])
        out.append((str(prepared.classes[c]), fpr, tpr, roc_auc_score(y == c, proba[:, c])))
    return out


def run_all_models(prepared, models=MODELS, trials=20, smote=True, out_dir="outputs", progress=None,
                   n_jobs=-1):
    """Tune, fit and compare `models`; save the best as one pipeline in `out_dir` (None: don't save).

    Models are ranked by cross-validated accuracy on the training rows. Returns (leaderboard,
    {name: fitted pipeline}, notes about skipped models).
    """
    rows, fitted, notes = [], {}, []
    n_classes = len(prepared.classes)
    for i, name in enumerate(models):
        if progress:
            progress(i, name)
        if name == "SVM" and len(prepared.y_train) > SVM_MAX_ROWS:
            notes.append(f"SVM skipped: {len(prepared.y_train):,} training rows, over {SVM_MAX_ROWS:,}.")
            continue
        start = time.time()
        try:
            params, cv_mean, cv_sd = tune(prepared, name, trials, smote, n_jobs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = pipeline(prepared, name, params, smote).fit(prepared.X_train, prepared.y_train)
        except Exception as e:  # a model that cannot run on this data is reported, not hidden
            notes.append(f"{name} failed: {e}")
            continue
        pred = model.predict(prepared.X_test)
        rows.append({
            "Model": name, "CV accuracy": cv_mean, "CV sd": cv_sd,
            "Held-out accuracy": accuracy_score(prepared.y_test, pred),
            "Held-out F1": f1_score(prepared.y_test, pred, average="weighted"),
            "Held-out ROC AUC": auc(model, prepared.X_test, prepared.y_test, n_classes),
            "Seconds": round(time.time() - start, 1), "Settings": params,
        })
        fitted[name] = model
    if progress:
        progress(len(models), None)
    if not rows:
        raise RuntimeError("No model could be trained: " + "; ".join(notes))
    board = pd.DataFrame(rows).sort_values("CV accuracy", ascending=False, kind="stable").reset_index(drop=True)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        board.drop(columns="Settings").to_csv(os.path.join(out_dir, "model_leaderboard.csv"), index=False)
        best = board.iloc[0]["Model"]
        save(fitted[best], prepared, best, os.path.join(out_dir, "best_model.pkl"))
    return board, fitted, notes


def save(model, prepared, name, path):
    """One file holding everything a prediction needs: the pipeline, its columns and the labels."""
    joblib.dump({"pipeline": model, "model": name, "target": prepared.target,
                 "features": prepared.features, "classes": prepared.classes}, path)


def predict(bundle, df):
    """Predicted labels for new rows. Columns are matched by name; extra ones, the target among
    them, are ignored, and a missing one is an error that names it."""
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"The file is missing columns the model was trained on: {', '.join(missing)}")
    codes = bundle["pipeline"].predict(df[bundle["features"]])
    return np.asarray(bundle["classes"], dtype=object)[codes]


def confusion(model, prepared):
    return confusion_matrix(prepared.y_test, model.predict(prepared.X_test), labels=range(len(prepared.classes)))
