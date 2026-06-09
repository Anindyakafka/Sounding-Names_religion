"""
Universal configuration for Sounding Names project.
Auto-detects environment (Google Colab vs local) and sets up paths accordingly.
"""

import os
import sys


def _detect_environment():
    """Detect whether we're running in Google Colab or locally."""
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


def get_base_dir(colab_path=None):
    """
    Return the project base directory regardless of environment.

    Parameters
    ----------
    colab_path : str, optional
        Path to use when running in Google Colab.
        Defaults to '/content/drive/My Drive/its_all_in_the_name_light_repo'.

    Returns
    -------
    str
        Absolute path to the project root.
    """
    env = _detect_environment()

    if env == "colab":
        base = colab_path or "/content/drive/My Drive/its_all_in_the_name_light_repo"
    else:
        # Assume this file lives in code/ → go one level up to project root
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    return base


def setup_directories(base_dir=None, colab_path=None):
    """
    Create and return all required project directories.

    Parameters
    ----------
    base_dir : str, optional
        Project root. Auto-detected if not provided.
    colab_path : str, optional
        Colab-specific path override.

    Returns
    -------
    dict
        Dictionary with keys 'base', 'data', 'models', 'output' mapping to paths.
    """
    if base_dir is None:
        base_dir = get_base_dir(colab_path)

    dirs = {
        "base": base_dir,
        "data": os.path.join(base_dir, "data"),
        "models": os.path.join(base_dir, "models"),
        "output": os.path.join(base_dir, "data", "predictions"),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs


def mount_drive_if_colab(force_remount=True):
    """Mount Google Drive only when running in Colab. No-op locally."""
    if _detect_environment() == "colab":
        from google.colab import drive
        drive.mount("/content/drive", force_remount=force_remount)
        print("Google Drive mounted.")
    else:
        print("Running locally — no drive mount needed.")


def install_requirements():
    """Install requirements.txt via pip (works in both Colab and local)."""
    base_dir = get_base_dir()
    req_path = os.path.join(base_dir, "code", "requirements.txt")
    if os.path.isfile(req_path):
        os.system(f"{sys.executable} -m pip install -r \"{req_path}\"")
        print(f"Installed requirements from {req_path}")
    else:
        print(f"Warning: requirements.txt not found at {req_path}")
