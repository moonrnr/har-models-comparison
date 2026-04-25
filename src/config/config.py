import os
import random
import numpy as np

from pathlib import Path

from src.config.constants import RANDOM_STATE


def find_project_root(start=Path.cwd()):
    for path in [start, *start.parents]:
        if (path / ".git").exists():
            return path
    raise RuntimeError("Could not find project root (.git)!")


def assert_exists(path: Path, msg: str = None):
    if not path.exists():
        raise FileNotFoundError(msg or f"Missing: {path}")


def validate_har_dataset(har_root: Path):
    assert_exists(har_root, f"Dataset not found: {har_root}")
    required_files = [
        har_root / "features.txt",
        har_root / "train" / "X_train.txt",
        har_root / "test" / "X_test.txt",
    ]
    for path in required_files:
        assert_exists(path)


def ensure_dir(path: Path):
    path.mkdir(exist_ok=True)
    return path


def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup():
    PROJECT_ROOT = find_project_root()

    DATA_DIR = PROJECT_ROOT / "data" / "human+activity+recognition+using+smartphones"
    HAR_ROOT = DATA_DIR / "UCI HAR Dataset"
    validate_har_dataset(HAR_ROOT)

    FIGURES_DIR = ensure_dir(PROJECT_ROOT / "figures")
    MODELS_DIR = ensure_dir(PROJECT_ROOT / "models")
    RESULTS_DIR = ensure_dir(PROJECT_ROOT / "results")

    set_random_seed(RANDOM_STATE)

    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "HAR_ROOT": HAR_ROOT,
        "RANDOM_STATE": RANDOM_STATE,
        "FIGURES_DIR": FIGURES_DIR,
        "MODELS_DIR": MODELS_DIR,
        "RESULTS_DIR": RESULTS_DIR,
    }
