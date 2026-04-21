from pathlib import Path
import os
import random
import numpy as np

def find_project_root(start=Path.cwd()):
    for path in [start, *start.parents]:
        if (path / ".git").exists():
            return path
    raise RuntimeError("Could not find project root (.git)!")

def setup():
    PROJECT_ROOT = find_project_root()
    DATA_DIR = PROJECT_ROOT / "data" / "human+activity+recognition+using+smartphones"
    HAR_ROOT = DATA_DIR / "UCI HAR Dataset"

    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

    if not HAR_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {HAR_ROOT}")

    if not (HAR_ROOT / "features.txt").exists():
        raise FileNotFoundError("Missing features.txt")

    if not (HAR_ROOT / "train" / "X_train.txt").exists():
        raise FileNotFoundError("Missing train/X_train.txt")

    if not (HAR_ROOT / "test" / "X_test.txt").exists():
        raise FileNotFoundError("Missing test/X_test.txt")

    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "HAR_ROOT": HAR_ROOT,
        "RANDOM_STATE": RANDOM_STATE,
    }
