# Sounding Names — Religion & Race Prediction from Names

A machine learning toolkit that predicts **religion** (Muslim vs. non-Muslim) and **race/ethnicity** (White, Black, Hispanic, Asian, Other) from personal names using character-level text features. Runs seamlessly in both Google Colab and local environments with automatic environment detection.

> ⚠️ **Ethical Notice:** This project is intended for academic research and audit purposes (e.g., studying bias in datasets). Predicting protected attributes from names carries significant ethical risks. Use responsibly and be aware of the limitations and potential harms of such inference.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [Religion Prediction](#religion-prediction)
  - [Race/Ethnicity Prediction](#raceethnicity-prediction)
- [Input Data Format](#input-data-format)
- [Output Format](#output-format)
- [Configuration Options](#configuration-options)
- [Model Files Reference](#model-files-reference)
- [Python Version Compatibility](#python-version-compatibility)
- [Dependencies](#dependencies)
- [License](#license)

---

## How It Works

The project contains two independent prediction pipelines:

### Religion Classification (`religion.ipynb`)

1. **Text Preprocessing** — Names are normalized using `unidecode` (transliterating Unicode to ASCII), uppercased, and cleaned of special characters via regex. Multiprocessing speeds up normalization on large datasets.
2. **Character n-gram TF-IDF Vectorization** — Cleaned names are wrapped with sentinel tokens (`{`, `}`) and spaces are replaced with `}{` to preserve word-boundary information at the character level. A pre-trained TF-IDF vectorizer converts these into sparse feature matrices.
3. **SVM Classification** — A pre-trained Support Vector Machine classifies each name as Muslim (`1`) or non-Muslim (`0`). The model outputs both a hard prediction and a continuous decision score (`muslim_score`), where higher values indicate greater likelihood of being Muslim.
4. **Concatenated Name Mode (optional)** — When `concat_model=True`, a parent/household member's name is concatenated with the individual's name (separated by `#` delimiters) before vectorization, allowing the model to leverage family-name context.
5. **Multiclass Mode (optional)** — When `n_way="multiclass"`, the model predicts across multiple religious categories instead of binary classification, using a label encoder stored in a pickle file.

### Race/Ethnicity Classification (`race.ipynb`)

1. **Text Preprocessing** — Similar normalization pipeline: `unidecode`, removal of digits and non-alphabetic characters, uppercasing, and whitespace collapsing.
2. **Character-Level Tokenization** — Names are converted to character sequences using a pre-trained Keras tokenizer, then padded/truncated to a fixed length.
3. **CNN Inference** — A pre-trained Convolutional Neural Network (Keras/TensorFlow) processes the character sequences. Two model variants exist:
   - **Text-only** (`cnn_USA_text.h5`) — Uses only name characters.
   - **Meta/GIS** (`cnn_USA_meta.h5`) — Augments name features with geographic/demographic prior probabilities (`pz_whi`, `pz_bla`, `pz_his`, `pz_asi`, `pz_oth`) for improved accuracy when census-tract-level data is available.
4. **5-Class Prediction** — Outputs probabilities for five categories: **A** (Asian), **B** (Black), **H** (Hispanic), **O** (Other), **W** (White), along with the argmax predicted label.

---

## Project Structure

```
Sounding-Names_religion/
├── code/
│   ├── config.py              # Universal environment detection & path setup
│   ├── religion.ipynb         # Religion prediction notebook (any Python 3.x)
│   ├── religion_py38.ipynb    # Religion notebook for Python 3.8 / TF 2.8 compat
│   ├── race.ipynb             # Race/ethnicity prediction notebook
│   ├── run_py38.py            # Standalone religion script (invoked by religion_py38)
│   └── requirements.txt       # Python dependencies
├── data/
│   ├── sample_data.csv              # Sample input for religion model
│   ├── sample_data_race.csv         # Sample input for race model
│   └── predictions/                 # Output directory (gitignored, auto-created)
├── models/                          # Pre-trained model files (included)
│   ├── *.sav                        # SVM models & TF-IDF vectorizers
│   ├── *.h5                         # CNN models (race)
│   └── *.pkl                        # Encoders & tokenizers
├── .gitignore                       # Ignores outputs, caches
├── PROJECT_LOG.md                   # Decision history & change rationale
├── LICENSE                          # MIT License
└── README.md
```

---

## Setup & Installation

Both notebooks use a shared `config.py` module that **auto-detects** whether you're running in Google Colab or locally. No manual path editing or commenting/uncommenting is needed — just run the cells as-is.

### Option A: Google Colab

1. Upload the entire repository to your Google Drive under `My Drive/its_all_in_the_name_light_repo/`.
2. Open the desired notebook in Google Colab.
3. Run all cells — the first cell auto-mounts Google Drive and installs dependencies.
4. If prompted, **restart the runtime** after dependency installation, then re-run all cells.

### Option B: Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd Sounding-Names_religion

# Create a Python 3.8–3.10 virtual environment (required for TF 2.8)
python3.10 -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r code/requirements.txt
pip install unidecode
```

Open either notebook in Jupyter Lab / VS Code and run all cells. Paths are resolved automatically relative to the notebook location.

### Custom Colab Path

If your repo lives at a non-default location in Google Drive, pass it explicitly:

```python
dirs = setup_directories(colab_path="/content/drive/My Drive/my_custom_path")
```

---

## Usage

### Religion Prediction

1. Place your CSV file in `data/` with at least a `name` column (and optionally a `parent` column for concatenated mode).
2. In `religion.ipynb` (or `religion_py38.ipynb`), set `dataname` to your filename.
3. Configure options (see [Configuration Options](#configuration-options)).
4. Run all cells. Results are saved to `data/predictions/<dataname>`.

**Standalone script:** You can also run `run_py38.py` directly without a notebook:
```bash
python3.8 code/run_py38.py
```

### Race/Ethnicity Prediction

1. Place your CSV file in `data/` with a `fullname` column (and optionally `pz_whi`, `pz_bla`, `pz_his`, `pz_asi`, `pz_oth` columns for GIS mode).
2. In `race.ipynb`, set `dataname` to your filename.
3. Set `usegis = True` if geographic priors are available in your data.
4. Run all cells. Results are saved to `data/predictions/<dataname>`.

---

## Input Data Format

### Religion Model (`sample_data.csv`)

| Column   | Description                              | Required |
|----------|------------------------------------------|----------|
| `name`   | Individual's full name                   | Yes      |
| `parent` | Parent/household member name (concat mode)| No       |

### Race Model (`sample_data_race.csv`)

| Column    | Description                                | Required        |
|-----------|--------------------------------------------|-----------------|
| `fullname`| Individual's full name                     | Yes             |
| `pz_whi`  | Prior probability of being White           | Only if `usegis=True` |
| `pz_bla`  | Prior probability of being Black           | Only if `usegis=True` |
| `pz_his`  | Prior probability of being Hispanic        | Only if `usegis=True` |
| `pz_asi`  | Prior probability of being Asian           | Only if `usegis=True` |
| `pz_oth`  | Prior probability of being Other           | Only if `usegis=True` |

---

## Output Format

### Religion Predictions

| Column              | Description                                          |
|---------------------|------------------------------------------------------|
| `name`              | Original name                                        |
| `predicted_religion`| Binary prediction (0 = non-Muslim, 1 = Muslim)       |
| `muslim_score`      | SVM decision function score (higher → more likely Muslim) |

### Race Predictions

| Column      | Description                                    |
|-------------|------------------------------------------------|
| `fullname`  | Original name                                  |
| `predicted` | Predicted ethnicity label (A/B/H/O/W)          |
| `A`         | Probability of Asian                           |
| `B`         | Probability of Black                           |
| `H`         | Probability of Hispanic                        |
| `O`         | Probability of Other                           |
| `W`         | Probability of White                           |

---

## Configuration Options

### Religion Notebook

| Variable       | Values                  | Description                                      |
|----------------|-------------------------|--------------------------------------------------|
| `name`         | Column name string      | Column containing the individual's name           |
| `pname`        | Column name string      | Column for parent/household member name           |
| `concat_model` | `True` / `False`        | Use concatenated name model                       |
| `n_way`        | `"2class"` / `"multiclass"` | Binary or multi-religion classification       |
| `classifier`   | `"svm"`                 | Classifier type (currently only SVM supported)    |

### Race Notebook

| Variable   | Values            | Description                                         |
|------------|-------------------|-----------------------------------------------------|
| `colname`  | Column name string| Column containing the full name                     |
| `usegis`   | `True` / `False`  | Include geographic/demographic prior probabilities  |
| `modelfile`| Filename string   | Auto-set based on `usegis`; selects CNN variant     |

---

## Model Files Reference

All model files are included in the `models/` directory. The naming convention encodes the configuration:

### Religion Models

| File | Mode | Concat | Purpose |
|------|------|--------|---------|
| `vectorizer_2class_svm_concat_False.sav` | Binary | No | TF-IDF vectorizer |
| `model_2class_svm_concat_False.sav` | Binary | No | SVM classifier |
| `vectorizer_2class_svm_concat_True.sav` | Binary | Yes | TF-IDF vectorizer |
| `model_2class_svm_concat_True.sav` | Binary | Yes | SVM classifier |
| `vectorizer_multiclass_svm_concat_False.sav` | Multi | No | TF-IDF vectorizer |
| `model_multiclass_svm_concat_False.sav` | Multi | No | SVM classifier |
| `vectorizer_multiclass_svm_concat_True.sav` | Multi | Yes | TF-IDF vectorizer |
| `model_multiclass_svm_concat_True.sav` | Multi | Yes | SVM classifier |
| `non_neural_label_encoding_multiclass.pkl` | Multi | — | Label ↔ ID mapping |
| `tokenizer_binary_False.pkl` | Binary | No | Character tokenizer |
| `tokenizer_binary_True.pkl` | Binary | Yes | Character tokenizer |
| `tokenizer_multiclass_False.pkl` | Multi | No | Character tokenizer |
| `tokenizer_multiclass_True.pkl` | Multi | Yes | Character tokenizer |

### Race/Ethnicity Models

| File | Purpose |
|------|---------|
| `cnn_USA_text.h5` | CNN text-only model |
| `cnn_USA_meta.h5` | CNN with GIS/demographic priors |
| `nc_voter_encoding.pkl` | Label encoder |
| `nc_voter_tokenizer.pkl` | Character-level tokenizer + vocab size + max length |

---

## Python Version Compatibility

| Component | Python 3.8–3.10 | Python 3.11+ |
|-----------|-----------------|--------------|
| Religion (`religion.ipynb`) | ✅ Full support | ✅ Works (no TF dependency) |
| Religion (`run_py38.py`) | ✅ Full support | ⚠️ May need older scikit-learn |
| Race (`race.ipynb`) | ✅ Full support | ❌ TF 2.8 incompatible |
| Race (TF 2.16+) | — | ✅ Upgrade `requirements.txt` |

> **Note:** TensorFlow 2.8.0 requires Python ≤ 3.10. The `religion_py38.ipynb` notebook installs Python 3.8 inside Colab to guarantee compatibility. For local use with Python 3.11+, consider upgrading to TensorFlow 2.16+ and adjusting `requirements.txt`.

---

## Dependencies

| Package        | Version  | Purpose                              |
|----------------|----------|--------------------------------------|
| tensorflow     | 2.8.0    | CNN model loading & inference (race) |
| keras          | 2.8.0    | Sequence preprocessing (race)        |
| scikit-learn   | 1.0.2    | SVM classifier & TF-IDF (religion)   |
| pandas         | 1.3.5    | Data loading & manipulation          |
| numpy          | 1.23.1   | Numerical operations                 |
| Unidecode      | latest   | Unicode-to-ASCII transliteration     |
| nltk           | 3.7      | Text processing utilities            |
| h5py           | 2.10.0   | HDF5 model file support              |

---

## License

This project is licensed under the [MIT License](LICENSE).
