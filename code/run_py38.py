"""
Religion prediction script (Python 3.8 compatible).
Designed to be invoked from religion_py38.ipynb via: !python3.8 run_py38.py
Can also be run directly: python run_py38.py
"""

import os
import sys
import pandas as pd
from unidecode import unidecode
from multiprocessing import Pool
import pickle
import warnings
warnings.filterwarnings("ignore")

# --- Universal path setup ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import setup_directories

dirs = setup_directories()
data_dir   = dirs["data"] + os.sep
model_dir  = dirs["models"] + os.sep
output_dir = dirs["output"] + os.sep

print(f"Base directory:   {dirs['base']}")
print(f"Data directory:   {data_dir}")
print(f"Model directory:  {model_dir}")
print(f"Output directory: {output_dir}")

# Name of data file
dataname = "sample_data.csv"

# --- Check Data ---
data = pd.read_csv(data_dir + dataname, encoding="utf-8")
print(f"Records: {len(data)}")
print(data.head())
del data

# --- Configuration ---
name = "name"       # column with individual's name
pname = "parent"    # column with parent/household member name
concat_model = False  # set True for concatenated names models
n_way = "2class"      # "2class" or "multiclass"
classifier = "svm"


def normalize(word):
    """Transliterate Unicode to ASCII."""
    if not isinstance(word, str):
        return ""
    return unidecode(word)


def _clean_name_series(series, n_workers=2):
    """Normalize, uppercase, and strip a pandas Series of names."""
    with Pool(n_workers) as p:
        cleaned = p.map(normalize, series)
    s = pd.Series(cleaned, index=series.index)
    s = s.str.upper()
    s = s.str.replace(r"[^A-Z .\-]", " ", regex=True)
    s = s.str.replace(r"[.\-]+", " ", regex=True)
    s = s.str.replace(r"([A-Z])\1{2,}", r"\1", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.strip()
    return s


def clean_data():
    """Read raw data, clean names, and save intermediate CSV."""
    data = pd.read_csv(data_dir + dataname, encoding="utf-8")
    data["name_cleaned"] = _clean_name_series(data[name], n_workers=2)

    if concat_model:
        data["pname_cleaned"] = _clean_name_series(
            data[pname], n_workers=min(10, os.cpu_count() or 2)
        )

    data.to_csv(output_dir + dataname, encoding="utf-8", index=False)
    print(f"Cleaned {len(data)} records -> {output_dir + dataname}")


def load_data():
    """Load cleaned data and apply character-level sentinel wrapping."""
    data = pd.read_csv(output_dir + dataname, encoding="utf-8")
    data["name_cleaned"] = data["name_cleaned"].fillna("")
    data["name_cleaned"] = "{" + data["name_cleaned"].str.replace(" ", "}{", regex=True) + "}"

    if concat_model:
        data["pname_cleaned"] = data["pname_cleaned"].fillna("")
        data["pname_cleaned"] = "{" + data["pname_cleaned"].str.replace(" ", "}{", regex=True) + "}"
        data["name_cleaned"] = "#" + data["name_cleaned"] + "#" + data["pname_cleaned"] + "#"

    return data


# --- Run Pipeline ---
clean_data()
data = load_data()
print(f"Loaded {len(data)} records for prediction.")

# Load pre-trained vectorizer and classifier
if n_way == "multiclass":
    label_path = model_dir + "non_neural_label_encoding_multiclass.pkl"
    assert os.path.isfile(label_path), f"Label encoding not found: {label_path}"
    with open(label_path, "rb") as f:
        (category_to_id, id_to_category) = pickle.load(f)

model_name = f"{n_way}_{classifier}_concat_{concat_model}.sav"
vec_path = model_dir + f"vectorizer_{model_name}"
clf_path = model_dir + f"model_{model_name}"
assert os.path.isfile(vec_path), f"Vectorizer not found: {vec_path}"
assert os.path.isfile(clf_path), f"Model not found: {clf_path}"

vectorizer = pickle.load(open(vec_path, "rb"))
clf = pickle.load(open(clf_path, "rb"))
print(f"Loaded model: {model_name}")

# Predict
tfidf_matrix = vectorizer.transform(data["name_cleaned"])
y_pred_prob = clf.decision_function(tfidf_matrix)
y_pred = clf.predict(tfidf_matrix)

if n_way == "2class":
    cols = {name: data[name], "predicted_religion": pd.Series(y_pred), "muslim_score": pd.Series(y_pred_prob)}
    if concat_model:
        cols[pname] = data[pname]
    df = pd.DataFrame(cols)
else:
    cols = {name: data[name], "predicted_religion": pd.Series(y_pred).map(id_to_category)}
    if concat_model:
        cols[pname] = data[pname]
    df = pd.DataFrame(cols)
    df2 = pd.DataFrame(y_pred_prob, columns=list(category_to_id.keys()))
    df = pd.concat([df.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)

out_path = output_dir + dataname
df.to_csv(out_path, encoding="utf-8", index=False)
print(f"Predictions saved -> {out_path}")
print(df.head())
