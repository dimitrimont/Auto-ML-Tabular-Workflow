# AutoML Tabular

A Streamlit app for classification on any CSV. Pick the file and the column to predict. The app reads
and prepares the data, tunes nine classifiers with Optuna, ranks them, and keeps the best one, with its
preprocessing, for predicting new rows.

Built in July 2025 and revised on 2026-10-18. The revision is described below, with what it was measured
against. The portfolio page runs the tuning both ways in the browser:
<https://dimitri-montgomery-site.pages.dev/ml/automl-tabular/>

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
pytest                      # 17 tests, under 30 s
```

The `data/` folder holds six public datasets for trying it:
- [Titanic](https://www.kaggle.com/c/titanic/data)
- [Adult income](https://archive.ics.uci.edu/dataset/2/adult)
- [Iris](https://archive.ics.uci.edu/dataset/53/iris)
- [Pima diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [red and white wine quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

## What it does

1. **Reads the CSV.** The separator (`,` `;` tab or `|`) and whether the first row is a header are
   detected, and both can be overridden. `?` is read as a missing value.
2. **Prepares the columns.**
   - Numbers have missing values filled with the mean and are then scaled.
   - Categories are one-hot encoded. Any seen fewer than 10 times are pooled into one column, and a
     missing value is its own category.
   - Identifiers (text or whole-number columns where every value is different, like a name or a row
     number) are left out. The app lists them, so you can put any of them back.
3. **Tunes nine classifiers.** These are RandomForest, XGBoost, LightGBM, CatBoost,
   LogisticRegression, KNN, SVM, ExtraTrees and NaiveBayes.
   - Each model gets 20 Optuna trials. Each trial is scored by 5-fold cross-validated accuracy on the
     training rows (80%).
   - Preprocessing, SMOTE and the model are one pipeline, fitted inside each fold, so no fold is
     scored by a model that saw its rows.
   - SVM is skipped above 10,000 training rows.
4. **Ranks them by cross-validation.** The held-out 20% is scored once, at the end, for every model.
   The scores are accuracy, weighted F1 and ROC AUC; with three or more classes, the AUC is each
   class against the rest. These numbers decide nothing. The best model is shown with its confusion
   matrix, its ROC curves (one per class when there are more than two) and a classification report.
5. **Predicts new rows.** The best pipeline is saved as `outputs/best_model.pkl`. Upload a CSV with
   the same columns, matched by name. Extra columns, the target among them, are ignored, and a
   missing one is named in the error. You can then download the predictions.

   The saved model is a pickle, so only load one this app wrote on your machine. Loading a pickle can
   run code.

`src/preprocess.py` holds the reading and preparation, and `src/models.py` the models, the tuning
and the comparison. `app.py` is the interface. `Phase_1.ipynb` is the exploration notebook from July
2025, kept as it ran. It needs the first version's modules, so check out the tag `first-version` to
run it.

## The revision

### Four of the six bundled datasets could not be run

Every dataset below was run through the first version as the app runs it: the file read with
`pd.read_csv` and the pipeline called with 20 trials. A fresh clone crashed every time, after
training all nine models, because the `outputs/` folder it saves to did not exist. The table shows runs
with that folder made by hand. Adult income ran with one trial per model instead of 20, and still trained for 28 minutes
before it crashed.

| Dataset | First version | Revised |
|---|---|---|
| Titanic | Ran, on 1,421 columns for 712 training rows: every name, ticket and cabin became a column | Runs on 14 columns |
| Pima diabetes | Ran | Runs |
| Iris | XGBoost failed on the text labels; the run then crashed after training (`multiclass format is not supported`) | Runs; ROC AUC averaged over the three classes |
| Red wine, white wine | Read as a single column (the files use `;`); crashed before training | Read as 12 columns; runs |
| Adult income | No header row, so the first person became the column names; XGBoost failed on the labels `' <=50K'` and `' >50K'`; after 28 minutes of training at one trial per model (the app runs 20), it crashed drawing the ROC curve (`pos_label is not specified`) | Runs: header added to the file, and the reader now notices a missing one |

XGBoost only accepts classes numbered from 0, so it would also have failed on the wine qualities (3
to 9) had the files been read. Each failure was caught and printed to the terminal, and the model was
left off the leaderboard without a word in the app.
Labels are now encoded for every model and decoded for every result shown.

Two smaller changes:
- A missing category used to be filled with the most common one, so the 687 Titanic passengers with
  no recorded cabin were all given cabin "G6". Missing is now its own category.
- The five models with no search space (LogisticRegression, KNN, SVM, ExtraTrees, NaiveBayes) ran
  20 identical trials each. Each now has a search space.

### The tuning scored each trial on rows the model had already seen

`objective()` fitted the model and returned its accuracy on the same rows. Memorising scores 1.000,
so the tuner could not tell overfitted settings apart:
- 180 of the 380 trials recorded in the notebook score exactly 1.000.
- Across 40 tunings measured here, the best score was exactly 1.000 in 35. All five exceptions were
  RandomForest on Titanic, which tops out near 0.95.

Each trial is now scored by 5-fold cross-validation. I measured both scores with the same search
spaces and 20 trials each: four tree models, two datasets, five random splits, judged on the held-out
rows.

| Held-out accuracy | Tuned on its own rows | Tuned by 5-fold CV | Difference |
|---|---|---|---|
| XGBoost, Titanic | 78.9% | 82.1% | +3.2 points, on all 5 splits |
| LightGBM, Titanic | 79.1% | 80.7% | +1.6 points, on all 5 splits |
| The other six model and dataset pairs | | | −0.6 to +1.0 points, no consistent direction |

Tree ensembles forgive overfitting, so on these datasets the pick changed less than the score
suggests. Cross-validation costs about 3.5 times as long per model: 109 s against 32 s on Titanic.

### The leaderboard ranked the models by the test rows

The first version ranked nine models by their accuracy on the 20% test split and reported the top
score as the winner's. On a few hundred rows that top score is partly luck. Now the models are ranked
by cross-validation, and the held-out rows are scored only after the choice.

To measure what that does, I needed a known answer. The 800-row synthetic datasets from scikit-learn's
`make_classification` give one, because each chosen model can also be scored on 50,000 fresh rows from
the same source. That fresh score is its true accuracy. Both versions ran on ten such datasets, each
with the same split:

| Ten synthetic datasets | First version | Revised |
|---|---|---|
| The leaderboard's number for the winner, mean | 85.1% | 84.3% (cross-validated) |
| The winner's true accuracy, mean | 84.0% | 83.9% |
| Number minus truth, mean | +1.1 points | +0.4 points |
| Number minus truth, mean size | 3.0 points | 0.8 points |
| Largest miss | 5.2 points | 3.4 points |

The picks were no better. The revised winner was 0.1 points less accurate on average: better on six
datasets, worse on four. What changed is the number the leaderboard reports. Cross-validation scores
all 640 training rows, not 160 test rows, so it lands much closer to the truth. Scored on its 160
held-out rows, the revised winner misses the truth by up to 5.5 points. That is why the held-out score
is shown but decides nothing.

### The notebook's last check ran on resampled rows

The notebook's final 10-fold cross-validation reported 0.843 accuracy and 0.928 ROC AUC for the best
model, an SVM on Titanic. It ran on the training rows after SMOTE. SMOTE builds each synthetic row
from a real row's neighbours, so a fold could be scored on rows made from its own training rows. The
same check gives different numbers:
- with SMOTE inside each fold: 0.827 and 0.872;
- on the held-out rows: 0.816 and 0.851.

## What is still open

- **Ranking metric.** Models are ranked by accuracy, while SMOTE trades accuracy for the smaller
  class's recall. On imbalanced data a balanced metric may be the better ranking.
- **SMOTE and one-hot columns.** SMOTE interpolates one-hot columns, so its synthetic rows can be
  0.4 of one category and 0.6 of another. SMOTENC would keep them whole.
- **Uncertainty.** Each result here is one run per split. The differences of a point or two between
  models on a few hundred held-out rows are within noise.

Built by Dimitri Montgomery with Streamlit, scikit-learn, Optuna, XGBoost, LightGBM, CatBoost and
imbalanced-learn.
