"""
AutoML for tabular data: pick a CSV and the column to predict, and the app prepares the data,
tunes nine classifiers by cross-validation, compares them, and keeps the best for predicting new
rows.

    streamlit run app.py
"""

import glob
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report

from src.models import FOLDS, MODELS, confusion, predict, roc_curves, run_all_models
from src.preprocess import Prepared, analyze_columns, load_csv

OUT = "outputs"
BEST = os.path.join(OUT, "best_model.pkl")
HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="AutoML Tabular", layout="wide")
st.title("AutoML Tabular")
st.caption(f"Nine classifiers, each tuned by {FOLDS}-fold cross-validation on the training rows and "
           "judged once on rows held out from everything.")

# ── 1. data ──────────────────────────────────────────────────────────────────────────────────

st.header("1. Train on your data")
samples = sorted(glob.glob(os.path.join(HERE, "data", "*.csv")))
where = st.radio("Data", ["Upload a CSV", "Use a sample from data/"], horizontal=True)
source = None
if where == "Upload a CSV":
    source = st.file_uploader("Choose a CSV file", type="csv")
else:
    source = st.selectbox("Sample", samples, format_func=os.path.basename)

if source is not None:
    if hasattr(source, "seek"):
        source.seek(0)
    df, sep, has_header = load_csv(source)
    keep = st.checkbox("The first row names the columns", value=has_header)
    if keep != has_header:
        if hasattr(source, "seek"):
            source.seek(0)
        df, sep, has_header = load_csv(source, header=keep)
    st.caption(f"Read {df.shape[0]:,} rows and {df.shape[1]} columns, separated by {sep!r}, "
               f"{'with' if has_header else 'without'} a header row.")
    st.dataframe(df.head())

    target = st.selectbox("Which column should the models predict?", df.columns, index=len(df.columns) - 1)
    identifiers = analyze_columns(df, target)[2]
    leave_out = st.multiselect(
        "Columns to leave out", [c for c in df.columns if c != target], default=identifiers,
        help="Identifiers, where every value is different, are left out by default: a model can only "
             "memorise them.")
    with st.expander("Settings"):
        trials = st.slider("Optuna trials per model", 5, 50, 20)
        smote = st.checkbox("Balance the classes with SMOTE (inside each fold)", value=True)
        chosen = st.multiselect("Models", MODELS, default=MODELS)

    if st.button("Run AutoML", type="primary", disabled=not chosen):
        prepared = Prepared(df, target, leave_out)
        if prepared.dropped_rows:
            st.info(f"{prepared.dropped_rows:,} rows have no {target!r} and are left out.")
        bar = st.progress(0.0)

        def progress(i, name):
            bar.progress(i / len(chosen), text=f"Tuning {name} ({i + 1} of {len(chosen)})" if name else "Done")

        try:
            board, fitted, notes = run_all_models(prepared, chosen, trials, smote, OUT, progress)
            st.session_state["result"] = {"board": board, "fitted": fitted, "notes": notes, "prepared": prepared}
        except Exception as e:
            st.error(f"The run stopped: {e}")

# ── results ──────────────────────────────────────────────────────────────────────────────────

result = st.session_state.get("result")
if result:
    board, prepared = result["board"], result["prepared"]
    st.subheader("Leaderboard")
    st.caption(f"Ranked by {FOLDS}-fold cross-validated accuracy on the {len(prepared.y_train):,} training "
               f"rows. The held-out columns score the {len(prepared.y_test):,} rows no choice was made on.")
    st.dataframe(board.drop(columns="Settings").style.format(precision=3), hide_index=True)
    for note in result["notes"]:
        st.warning(note)

    best = board.iloc[0]["Model"]
    model = result["fitted"][best]
    labels = [str(c) for c in prepared.classes]
    st.subheader(f"Best: {best}")
    st.caption(f"Settings: {board.iloc[0]['Settings']}. Saved with its preprocessing to {BEST}.")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion(model, prepared), annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set(title="Held-out rows", xlabel="Predicted", ylabel="Actual")
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots(figsize=(5, 4))
        for label, fpr, tpr, area in roc_curves(model, prepared):
            ax.plot(fpr, tpr, label=f"{label} (AUC {area:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(title="ROC, held-out rows", xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(fontsize=8)
        st.pyplot(fig)

    report = classification_report(prepared.y_test, model.predict(prepared.X_test), labels=range(len(labels)),
                                   target_names=labels, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().style.format(precision=3))

# ── 2. predict ───────────────────────────────────────────────────────────────────────────────

st.header("2. Predict new rows")
new = st.file_uploader("A CSV with the same columns (the target column may be left out)", type="csv", key="new")
if not os.path.exists(BEST):
    st.info("Train a model in step 1 first; the best one is kept for predictions.")
elif new is not None:
    bundle = joblib.load(BEST)  # written by step 1 on this machine; never load someone else's .pkl
    rows, _, _ = load_csv(new)
    try:
        rows[f"Predicted {bundle['target']}"] = predict(bundle, rows)
    except ValueError as e:
        st.error(str(e))
    else:
        st.caption(f"{len(rows):,} rows predicted by {bundle['model']}.")
        st.dataframe(rows)
        st.download_button("Download predictions as CSV", rows.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")
