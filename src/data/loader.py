import pandas as pd
from pathlib import Path


def _fix_duplicate_feature_names(names):
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def load_har_data(har_root: Path | None = None):
    if har_root is None:
        raise ValueError("HAR_ROOT path must be provided")

    features = pd.read_csv(
        har_root / "features.txt", sep=r"\s+", header=None, names=["idx", "name"]
    )
    feature_names = _fix_duplicate_feature_names(features["name"].tolist())

    activities = pd.read_csv(
        har_root / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "label"]
    )
    activity_map = dict(zip(activities["id"], activities["label"]))

    X_train = pd.read_csv(
        har_root / "train" / "X_train.txt", sep=r"\s+", header=None, names=feature_names
    )
    y_train = pd.read_csv(
        har_root / "train" / "y_train.txt",
        sep=r"\s+",
        header=None,
        names=["activity_id"],
    )["activity_id"]
    y_train = y_train.map(activity_map).astype("category")
    groups_train = pd.read_csv(
        har_root / "train" / "subject_train.txt",
        sep=r"\s+",
        header=None,
        names=["subject"],
    )["subject"]

    X_test = pd.read_csv(
        har_root / "test" / "X_test.txt", sep=r"\s+", header=None, names=feature_names
    )
    y_test = pd.read_csv(
        har_root / "test" / "y_test.txt", sep=r"\s+", header=None, names=["activity_id"]
    )["activity_id"]
    y_test = y_test.map(activity_map).astype("category")
    groups_test = pd.read_csv(
        har_root / "test" / "subject_test.txt",
        sep=r"\s+",
        header=None,
        names=["subject"],
    )["subject"]

    return X_train, y_train, groups_train, X_test, y_test, groups_test, feature_names
