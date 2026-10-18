"""
Reading any CSV and turning its columns into numbers a model can use.

Revised 2026-10-18:

- The wine files separate values with ";" and were read as one column; the adult-income file has
  no header row, so its first person became the column names. `load_csv` now detects the
  separator and whether the first row is a header, and reads "?" as missing.
- Every name, ticket and cabin in the Titanic file became its own one-hot column: 1,421 columns
  for 712 training rows. Columns that are identifiers (every value different) are now dropped,
  and categories seen fewer than 10 times are pooled into one column.
- A missing category was filled with the most common one, so 687 Titanic passengers with no
  recorded cabin were all given cabin "G6". It is now its own category, "missing".
- Labels are encoded as 0..n-1 (XGBoost rejects anything else, so it failed on "setosa" and
  " >50K", and would have on wine qualities 3..9) and decoded again for every result shown.
- The preprocessing is returned unfitted, to be fitted inside each cross-validation fold.
"""

import csv
import io

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

MISSING = ["?", ""]  # read as missing, besides pandas' own markers
RARE = 10  # a category seen fewer times than this is pooled with the other rare ones


def _numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def first_row_is_header(rows):
    """Whether the first of these rows names the columns.

    It does when every column says so: over a column of numbers, the first cell is not a number;
    over a column of text, the first cell never appears again below it.
    """
    head, body = rows[0], rows[1:]
    if not body:
        return True
    for i, cell in enumerate(head):
        below = [r[i].strip() for r in body if i < len(r) and r[i].strip() not in MISSING]
        if below and all(_numeric(v) for v in below):
            if _numeric(cell.strip()):
                return False
        elif cell.strip() in below:
            return False
    return True


def sniff(text):
    """(separator, has_header) for the start of a CSV file."""
    try:
        sep = csv.Sniffer().sniff(text, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=sep) if r][:200]
    if len(rows) > 2:  # the last row of the sample may be cut short
        rows = rows[:-1]
    return sep, first_row_is_header(rows)


def load_csv(source, header=None):
    """A DataFrame from a path or an uploaded file, with the separator and header row detected.

    `header` overrides the detection. A file without a header gets columns column_1, column_2...
    Returns (DataFrame, separator, has_header).
    """
    raw = source.read() if hasattr(source, "read") else open(source, "rb").read()
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    sep, detected = sniff(text[:65536])
    has_header = detected if header is None else header
    df = pd.read_csv(io.StringIO(text), sep=sep, header=0 if has_header else None,
                     skipinitialspace=True, na_values=MISSING)
    if not has_header:
        df.columns = [f"column_{i + 1}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    return df, sep, has_header


def analyze_columns(df, target_col):
    """(numeric columns, categorical columns, dropped identifiers), the target left out.

    An identifier is a text or whole-number column whose values are all different, such as a
    passenger's name or a row number: it cannot repeat, so a model can only memorise it.
    """
    numeric, categorical, identifiers = [], [], []
    for col in df.columns:
        if col == target_col:
            continue
        values = df[col].dropna()
        is_number = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
        whole = is_number and len(values) and np.all(np.mod(values, 1) == 0)
        if len(values) > 20 and values.nunique() == len(values) and (not is_number or whole):
            identifiers.append(col)
        elif is_number:
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical, identifiers


def make_preprocessor(numeric, categorical):
    """An unfitted ColumnTransformer: numbers imputed and scaled, categories one-hot encoded."""
    num = Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=RARE,
                                  sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


class Prepared:
    """A dataset split for modelling, with what was done to it."""

    def __init__(self, df, target_col, leave_out=None, test_size=0.2, random_state=42):
        """`leave_out` lists columns not to use; by default, the identifiers found."""
        self.target = target_col
        self.dropped_rows = int(df[target_col].isna().sum())  # rows with no answer can't teach
        df = df.dropna(subset=[target_col])
        numeric, categorical, self.identifiers = analyze_columns(df, target_col)
        self.left_out = self.identifiers if leave_out is None else [c for c in leave_out if c != target_col]
        if leave_out is not None:  # an identifier the user chose to keep goes back in as its kind
            numeric = [c for c in numeric + self.identifiers if pd.api.types.is_numeric_dtype(df[c])]
            categorical = [c for c in categorical + self.identifiers if c not in numeric]
        self.numeric = [c for c in df.columns if c in numeric and c not in self.left_out]
        self.categorical = [c for c in df.columns if c in categorical and c not in self.left_out]
        self.features = self.numeric + self.categorical
        self.encoder = LabelEncoder().fit(df[target_col])
        self.classes = list(self.encoder.classes_)
        X, y = df[self.features], self.encoder.transform(df[target_col])
        counts = np.bincount(y)
        stratify = y if counts.min() >= 2 else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    def preprocessor(self):
        return make_preprocessor(self.numeric, self.categorical)

    def width(self):
        """How many columns the models see once the training rows are encoded."""
        return self.preprocessor().fit_transform(self.X_train).shape[1]

    def decode(self, y):
        return self.encoder.inverse_transform(np.asarray(y))
