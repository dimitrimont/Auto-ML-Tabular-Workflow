"""
Known-answer tests for the pipeline: run with `pytest` from the repository root.

The models run with few trials and one process so the suite stays quick; what is tested is that
the pipeline reads, prepares, scores and saves correctly, not how good the models are.
"""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models import auc, predict, roc_curves, run_all_models, smote_neighbours, tune
from src.preprocess import Prepared, load_csv

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FAST = ["LogisticRegression", "LightGBM"]
pytestmark = pytest.mark.filterwarnings("ignore")


def data(name):
    return load_csv(os.path.join(DATA, name))[0]


def test_reads_semicolon_separated_files():
    df, sep, header = load_csv(os.path.join(DATA, "red_wine_quality.csv"))
    assert (sep, header, df.shape) == (";", True, (1599, 12))
    assert df.columns[-1] == "quality"


def test_notices_a_file_without_a_header(tmp_path):
    lines = open(os.path.join(DATA, "adult_income.csv")).read().splitlines()
    (tmp_path / "no_header.csv").write_text("\n".join(lines[1:51]) + "\n")
    df, sep, header = load_csv(str(tmp_path / "no_header.csv"))
    assert (sep, header, df.shape) == (",", False, (50, 15))
    assert df.columns[0] == "column_1" and df.iloc[0, 0] == 39  # the first person is a row, not names
    df, _, header = load_csv(os.path.join(DATA, "adult_income.csv"))
    assert header and df.shape == (32561, 15)


def test_question_marks_are_missing_values():
    df = data("adult_income.csv")
    assert df["workclass"].isna().sum() == 1836  # written " ?" in the file
    assert "?" not in set(df["workclass"].dropna())


def test_identifiers_are_left_out_and_rare_categories_pooled():
    p = Prepared(data("titanic.csv"), "Survived")
    assert p.identifiers == ["PassengerId", "Name"]
    assert p.width() < 20  # was 1,421 columns for 712 rows


def test_a_missing_category_stays_missing():
    p = Prepared(data("titanic.csv"), "Survived")
    encoder = p.preprocessor().fit(p.X_train).named_transformers_["cat"].named_steps["encoder"]
    cabins = encoder.categories_[p.categorical.index("Cabin")]
    assert "missing" in cabins  # not 687 passengers given cabin "G6"


def test_labels_any_model_can_use():
    p = Prepared(data("iris_flower.csv"), "species")
    assert p.classes == ["setosa", "versicolor", "virginica"]
    board, fitted, notes = run_all_models(p, ["XGBoost"], trials=2, out_dir=None, n_jobs=1)
    assert notes == [] and board.iloc[0]["Held-out accuracy"] > 0.85  # XGBoost rejected these labels
    assert set(p.decode(fitted["XGBoost"].predict(p.X_test))) <= set(p.classes)
    wine = Prepared(data("red_wine_quality.csv"), "quality")
    assert wine.classes == [3, 4, 5, 6, 7, 8] and wine.decode([0])[0] == 3


def test_tuning_is_not_fooled_by_labels_that_are_noise():
    """Known answer: when the label is a coin toss, nothing can be predicted, so an honest tuning
    score is near 0.5. Training accuracy, the first version's score, says nearly 1.0."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(300, 5)), columns=list("abcde"))
    df["label"] = rng.integers(0, 2, size=300)
    p = Prepared(df, "label")
    deep = RandomForestClassifier(max_depth=20, random_state=42).fit(p.X_train, p.y_train)
    assert deep.score(p.X_train, p.y_train) > 0.95
    _, cv_accuracy, _ = tune(p, "RandomForest", trials=5, n_jobs=1)
    assert 0.35 < cv_accuracy < 0.65


def test_auc_and_curves_for_several_classes():
    p = Prepared(data("iris_flower.csv"), "species")
    board, fitted, _ = run_all_models(p, ["LogisticRegression"], trials=2, out_dir=None, n_jobs=1)
    model = fitted["LogisticRegression"]
    assert auc(model, p.X_test, p.y_test, 3) > 0.9  # was reported as 0.0, then crashed the run
    assert [c[0] for c in roc_curves(model, p)] == p.classes


def test_curve_for_two_text_labels():
    p = Prepared(data("adult_income.csv").iloc[:3000], "income")
    _, fitted, _ = run_all_models(p, ["LogisticRegression"], trials=2, out_dir=None, n_jobs=1)
    curves = roc_curves(fitted["LogisticRegression"], p)
    assert len(curves) == 1 and curves[0][0] == ">50K" and curves[0][3] > 0.8


def test_smote_neighbours_fit_the_smallest_class():
    assert smote_neighbours(np.array([0] * 100 + [1] * 100)) == 5
    assert smote_neighbours(np.array([0] * 100 + [1] * 5)) == 3  # 4 in a fold's training part
    assert smote_neighbours(np.array([0] * 100 + [1] * 2)) == 0  # too few: SMOTE is skipped


@pytest.mark.parametrize("name, target", [
    ("adult_income.csv", "income"), ("diabetes.csv", "Outcome"), ("iris_flower.csv", "species"),
    ("red_wine_quality.csv", "quality"), ("titanic.csv", "Survived"), ("white_wine_quality.csv", "quality")])
def test_every_bundled_dataset_runs(name, target, tmp_path):
    df = data(name)
    if len(df) > 4000:
        df = df.sample(4000, random_state=0)
    board, _, notes = run_all_models(Prepared(df, target), FAST, trials=2, out_dir=str(tmp_path), n_jobs=1)
    assert notes == [] and list(board["Model"]) != []
    assert (tmp_path / "best_model.pkl").exists()


def test_saves_one_file_that_predicts_new_rows(tmp_path):
    out = tmp_path / "outputs"  # does not exist yet; the first version crashed here
    df = data("titanic.csv")
    p = Prepared(df, "Survived")
    run_all_models(p, ["LogisticRegression"], trials=2, out_dir=str(out), n_jobs=1)
    import joblib
    bundle = joblib.load(out / "best_model.pkl")  # the file this test just wrote
    new = df.iloc[:25]  # still has the Survived column, which is ignored
    assert set(predict(bundle, new)) <= {0, 1}
    with pytest.raises(ValueError, match="Sex"):
        predict(bundle, new.drop(columns="Sex"))
