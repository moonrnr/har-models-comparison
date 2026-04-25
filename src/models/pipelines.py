from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from src.config.constants import RANDOM_STATE, N_JOBS


def pipe_dummy(strategy="stratified"):
    return Pipeline(
        [
            ("clf", DummyClassifier(strategy=strategy, random_state=RANDOM_STATE)),
        ]
    )


def pipe_logreg():
    return Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("selector", "passthrough"),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def pipe_linear_svc():
    return Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("selector", "passthrough"),
            (
                "clf",
                LinearSVC(
                    max_iter=10000,
                    dual="auto",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def pipe_rbf_svc(probability=False):
    return Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("selector", "passthrough"),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    probability=probability,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def pipe_random_forest():
    return Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def pipe_gaussian_nb():
    return Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("selector", "passthrough"),
            ("clf", GaussianNB()),
        ]
    )
