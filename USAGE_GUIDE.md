# Usage Guide — Sounding Names

A step-by-step guide for using this repository to predict religion and race/ethnicity from personal names.

---

## Credits

This project is based on the original codebase by **Sugat Chaturvedi**. The initial models, data preprocessing pipeline, and prediction logic were developed as part of their research on name-based demographic inference. This fork adds universal environment detection, Python 3.8 compatibility tooling, improved error handling, and comprehensive documentation while preserving the original model architecture and weights.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8, 3.9, or 3.10 (required for TensorFlow 2.8) |
| **OS** | Windows, macOS, or Linux |
| **RAM** | ≥ 4 GB (model loading + inference) |
| **Disk** | ~500 MB (models + dependencies) |

> ⚠️ **Python 3.11+ is NOT supported** for the race pipeline. TensorFlow 2.8.0 does not support Python ≥ 3.11. The religion pipeline may load models on newer Python but will produce incorrect results due to scikit-learn version incompatibility (models were trained with sklearn 0.22.2). **Use Python 3.8–3.10 for correct predictions.**

---

## Quick Start (5 Minutes)

### Step 1: Clone & Set Up Environment

```bash
git clone <repo-url>
cd Sounding-Names_religion

# Create a virtual environment with Python 3.10 (recommended)
python3.10 -m venv .venv

# Activate it
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r code/requirements.txt
pip install unidecode
```

### Step 2: Run Religion Prediction

Open `code/religion.ipynb` in Jupyter Lab or VS Code and run all cells.

Or use the standalone script:
```bash
python code/run_py38.py
```

Results are saved to `data/predictions/sample_data.csv`.

### Step 3: Run Race/Ethnicity Prediction

Open `code/race.ipynb` and run all cells.

Results are saved to `data/predictions/sample_data_race.csv`.

---

## Using Your Own Data

### Religion Prediction

1. Create a CSV file with at least a `name` column:
   ```csv
   name,parent
   Aisha Khan,Rashid Khan
   John Smith,Robert Smith
   Priya Sharma,Amit Sharma
   ```

2. Place it in `data/` (e.g., `data/my_names.csv`).

3. In `religion.ipynb`, change:
   ```python
   dataname = "my_names.csv"
   ```

4. Optionally enable concatenated name mode for better accuracy when parent names are available:
   ```python
   concat_model = True
   ```

5. Run all cells. Output appears in `data/predictions/my_names.csv`.

### Race/Ethnicity Prediction

1. Create a CSV with a `fullname` column:
   ```csv
   fullname
   JEFFREY TODD BUCKNER
   MARIA GARCIA RODRIGUEZ
   WEI CHEN
   ```

2. Place it in `data/` and update `dataname` in `race.ipynb`.

3. If you have census-tract demographic priors, add columns `pz_whi`, `pz_bla`, `pz_his`, `pz_asi`, `pz_oth` and set:
   ```python
   usegis = True
   ```

4. Run all cells.

---

## Google Colab Setup

If you don't have Python 3.8–3.10 locally, use Google Colab:

1. Upload the entire repository to Google Drive at:
   ```
   My Drive/its_all_in_the_name_light_repo/
   ```

2. Open any notebook in Colab.

3. Run all cells — the first cell auto-detects Colab, mounts Drive, and installs dependencies.

4. For `religion_py38.ipynb`: this notebook installs Python 3.8 inside Colab to guarantee TF 2.8 compatibility. Run cells sequentially and wait for each to complete.

---

## Understanding the Output

### Religion Predictions

| Column | Meaning |
|--------|---------|
| `predicted_religion` | `0` = non-Muslim, `1` = Muslim |
| `muslim_score` | Continuous SVM decision score. Higher = more likely Muslim. Negative = more likely non-Muslim. Use this for ranking/thresholding rather than the binary label alone. |

### Race Predictions

| Column | Meaning |
|--------|---------|
| `predicted` | Most likely category: `W` (White), `B` (Black), `H` (Hispanic), `A` (Asian), `O` (Other) |
| `W`, `B`, `H`, `A`, `O` | Probability scores for each category (sum ≈ 1.0). Use these for nuanced analysis rather than the hard label. |

---

## Configuration Reference

### Religion Notebook (`religion.ipynb`)

```python
name = "name"           # Column with person's name
pname = "parent"        # Column with parent/household member name
concat_model = False    # True = use parent name for better accuracy
n_way = "2class"        # "2class" (Muslim/non-Muslim) or "multiclass"
classifier = "svm"      # Only SVM is currently supported
```

### Race Notebook (`race.ipynb`)

```python
colname = "fullname"    # Column with person's full name
usegis = False          # True = use demographic prior probabilities
```

---

## Available Models

The `models/` directory contains pre-trained models for every configuration combination:

| Model Variant | Binary (2class) | Multiclass | Single Name | Concatenated |
|---------------|:-:|:-:|:-:|:-:|
| Religion SVM | ✅ | ✅ | ✅ | ✅ |
| Race CNN (text) | — | ✅ (5-class) | ✅ | — |
| Race CNN (GIS) | — | ✅ (5-class) | ✅ | — |

Model naming convention: `{mode}_{classifier}_concat_{True|False}.sav`

---

## Troubleshooting

### "NotFittedError: TfidfTransformer instance is not fitted"
Your scikit-learn version is too new. The models were trained with sklearn 0.22.2. Use `scikit-learn==1.0.2` (as pinned in `requirements.txt`) with Python 3.8–3.10.

### "ModuleNotFoundError: No module named 'imp'"
TensorFlow 2.8.0 is incompatible with Python 3.12+. Use Python 3.8–3.10, or use `religion_py38.ipynb` which installs Python 3.8 inside Colab.

### "Model not found: /path/to/models/..."
Ensure all model files are present in the `models/` directory. See `models/.gitkeep` for the complete list of 17 required files.

### Predictions look wrong
- Verify you're using Python 3.8–3.10 with the exact dependency versions in `requirements.txt`.
- Check that input names are in the expected column (`name` for religion, `fullname` for race).
- Ensure Unicode names are properly encoded (UTF-8 CSV).

---

## Ethical Considerations

> ⚠️ This tool infers protected attributes from names. Such inference is inherently imperfect and carries risks of misclassification, stereotyping, and discriminatory application.

- **Accuracy varies** significantly across name origins, transliterations, and cultural contexts.
- **Never use** for individual-level decisions (hiring, lending, law enforcement, etc.).
- **Appropriate uses**: aggregate-level bias audits, demographic research with IRB approval, studying representation gaps in datasets.
- **Always report** confidence intervals and acknowledge limitations when publishing results.

---

## License

MIT License — see [LICENSE](LICENSE).
