# Project Log

Chronological record of decisions, changes, and rationale for the Sounding Names project.

---

## 2026-06-09 — Initial README Rewrite

**What:** Replaced the one-line README with a comprehensive documentation file.

**Why:** The original README (`"A project to detect religion from sounding names"`) provided no guidance on setup, usage, data formats, or how the models work. New users had to reverse-engineer everything from notebook code.

**Decisions:**
- Added ethical notice upfront — predicting protected attributes from names carries real harm potential; users must be aware before running anything.
- Documented both pipelines (religion SVM + race CNN) with step-by-step technical explanations so researchers understand what each stage does.
- Included input/output schema tables derived from actual sample CSV files, since these were never documented.
- Listed all configuration variables with their valid values and effects.

---

## 2026-06-09 — Universal Environment Detection (`config.py`)

**What:** Created `code/config.py` — a shared module that auto-detects Google Colab vs. local execution and configures all paths accordingly. Updated both notebooks to use it.

**Why:** Both notebooks had hardcoded Google Drive paths (`/content/drive/My Drive/its_all_in_the_name_light_repo`) as the active `base_dir`, with the local path commented out. Every user had to manually comment/uncomment lines depending on their environment. This was fragile and error-prone.

**Decisions:**
- Used `import google.colab` try/except for detection rather than environment variables — more reliable since Colab always has this module and local never does.
- Local paths resolve relative to `config.py`'s own location (`code/` → parent = project root), making the repo portable regardless of where it's cloned.
- `os.makedirs(exist_ok=True)` replaces manual `if not os.path.isdir` checks — cleaner and race-condition safe.
- `sys.executable` used for pip invocation instead of bare `!pip` — works outside IPython/Colab.
- Kept `colab_path` as an optional override parameter for non-default Drive locations.

---

## 2026-06-09 — Notebook Code Quality Improvements

**What:** Refactored data cleaning, model loading, and prediction cells in both `religion.ipynb` and `race.ipynb`.

**Why:** Original code had several issues: duplicated cleaning logic, unsafe Pool usage without context managers, no error messages when model files were missing, deprecated pandas patterns (`inplace=True` on chained operations), and no feedback during execution.

**Decisions:**
- **Religion notebook:** Extracted `_clean_name_series()` helper to eliminate copy-pasted regex chains between `name` and `pname` cleaning. Added `isinstance(word, str)` guard in `normalize()` to handle NaN/numeric values gracefully.
- **Race notebook:** Replaced hardcoded `Pool(4)` / `Pool(10)` with `min(N, os.cpu_count())` to avoid spawning more workers than cores. Used `with Pool(...) as p:` context manager to prevent zombie processes.
- **Both notebooks:** Added `assert os.path.isfile(...)` with descriptive messages before every model/pickle load. Previously, a missing model file produced a cryptic `FileNotFoundError` with no indication of *which* file or *where* to put it.
- **Both notebooks:** Added print statements at each stage (record counts, output paths, model names) so users can verify progress in long-running pipelines.
- **Race notebook:** Consolidated five identical `default_*_prob = 1/5` variables into a single `default_prob`.

---

## 2026-06-09 — Repository Restructuring

**What:** Added `.gitignore`, created `models/` directory with `.gitkeep` and manifest, added `PROJECT_LOG.md`.

**Why:** The repo had no `.gitignore`, meaning prediction outputs, `__pycache__`, and potentially large model files could be accidentally committed. The `models/` directory was referenced in code but didn't exist in the repo, causing immediate failures for new clones. There was no record of why decisions were made.

**Decisions:**
- **`.gitignore`:** Ignores `data/predictions/*.csv` (regenerable outputs), `models/*.sav|*.h5|*.pkl` (large binaries), `__pycache__/`, `.ipynb_checkpoints/`, and IDE/OS artifacts. Sample *input* CSVs in `data/` are intentionally tracked since they serve as documentation.
- **`models/.gitkeep`:** Ensures the directory exists in git while keeping actual model files out. Includes a manifest comment listing every expected file and which notebook needs it, so users know exactly what to obtain.
- **`PROJECT_LOG.md`:** This file. Ascending chronological order so the most recent context is at the bottom (closest to where you're reading). Each entry records *what* changed, *why*, and the specific *decisions* made — not just a changelog but the reasoning behind choices.
- **Kept flat structure** (`code/`, `data/`, `models/` at root): The project has only two notebooks and one shared module. Adding `src/`, `notebooks/`, `tests/` etc. would be over-engineering for this scale. If the project grows (e.g., training scripts, evaluation metrics, API wrapper), a deeper structure would be warranted then.

---

## 2026-06-09 — Added Python 3.8 Compatibility Layer & Model Files

**What:** User added `religion_py38.ipynb`, `run_py38.py`, and all pre-trained model files to `models/`. Refactored both new files to use universal `config.py` instead of hardcoded G-Drive paths. Updated README with model file reference table and Python version compatibility matrix.

**Why:** TensorFlow 2.8.0 (required by the race CNN models) does not support Python 3.11+. The `religion_py38.ipynb` notebook installs Python 3.8 inside Colab as a workaround. The standalone `run_py38.py` script allows running religion prediction without a notebook. All 17 model files were added to the repo, making it fully self-contained.

**Decisions:**
- **`run_py38.py` refactored** to import `config.setup_directories()` instead of hardcoding `/content/drive/My Drive/...`. Now works identically in Colab and local without any edits. Also received the same code quality improvements as `religion.ipynb` (context-managed Pool, `_clean_name_series` helper, assert-based model loading).
- **`religion_py38.ipynb` updated**: Replaced the `%%writefile run_py38.py` cell (which generated the script inline with old hardcoded paths) with a markdown explanation + direct invocation of the now-standalone `run_py38.py`. Install cell uses `config.get_base_dir()` to locate `requirements.txt` regardless of environment.
- **`models/.gitkeep` manifest expanded** from 5 files to all 17 actual model files, organized by pipeline (religion vs race) with mode/concat annotations matching the naming convention.
- **README gained two new sections**: "Model Files Reference" (complete table of all 17 files with mode/concat/purpose) and "Python Version Compatibility" (matrix showing which components work on which Python versions, with upgrade guidance for TF 2.16+).
- **Removed "models not included" note** from README since all models are now present in the repository.

---

## 2026-06-09 — Pipeline Verification, Usage Guide & Credits

**What:** Ran end-to-end verification of both prediction pipelines. Created `USAGE_GUIDE.md`. Added credits to Sugat Chaturvedi in README and guide. Corrected Python version compatibility table based on actual test results.

**Why:** User asked whether the code actually works and requested a beginner-friendly guide with proper attribution.

**Verification Results:**
- **Religion pipeline on Python 3.13 + sklearn 1.6.1: FAILS.** Models were trained with sklearn 0.22.2. When unpickled with sklearn 1.6.1, the internal `TfidfTransformer` loses its `idf_` attribute, causing `NotFittedError` at transform time. The version mismatch warning is not just cosmetic — it produces broken inference.
- **Race pipeline on Python 3.13: FAILS.** TensorFlow 2.8.0 uses the removed `imp` module and cannot import at all on Python ≥ 3.12.
- **Both pipelines on Python 3.8–3.10: EXPECTED TO WORK.** Dependency versions in `requirements.txt` are compatible with this range. Not tested locally due to absence of a py3.10 venv, but all model files load correctly and config path resolution is verified.
- **Config module: WORKS** on any Python version. Path resolution, directory creation, and environment detection all function correctly.

**Decisions:**
- **Corrected compatibility table** in README: previously listed religion as "✅ Works (no TF dependency)" on Python 3.11+, which was wrong. Updated to "❌ sklearn version mismatch" for both religion components.
- **Created `USAGE_GUIDE.md`** as a separate document rather than expanding README further. Covers: prerequisites, quick start, using custom data, Colab setup, output interpretation, configuration reference, model inventory, troubleshooting (with specific error messages and fixes), and ethical considerations.
- **Added credits** to Sugat Chaturvedi in both README (new Credits section after ethical notice) and USAGE_GUIDE (top of document). Acknowledges original codebase, models, and preprocessing pipeline while noting what this fork adds.
- **Linked USAGE_GUIDE from README** via new Table of Contents entry and dedicated section before License.
