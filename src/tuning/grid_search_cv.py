import pandas as pd
import time
import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV


def run_grid(
    name,
    factory,
    param_grid,
    *,
    X,
    y,
    groups,
    models_dir,
    results_dir,
    cv,
    scoring,
    n_jobs,
    verbose=1,
):
    if not Path(models_dir).exists():
        raise FileNotFoundError(f"models_dir nie istnieje: {models_dir}")

    if not Path(results_dir).exists():
        raise FileNotFoundError(f"results_dir nie istnieje: {results_dir}")

    grid = GridSearchCV(
        estimator=factory(),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=True,
        n_jobs=n_jobs,
        return_train_score=True,
        verbose=verbose,
    )

    t0 = time.perf_counter()
    grid.fit(X, y, groups=groups)
    elapsed = time.perf_counter() - t0

    grid_path = models_dir / f"grid_{name}.joblib"
    csv_path = results_dir / f"cv_{name}.csv"
    joblib.dump(grid, grid_path)
    pd.DataFrame(grid.cv_results_).to_csv(csv_path, index=False)

    n_combos = len(grid.cv_results_["params"])
    n_fits = n_combos * cv.get_n_splits()
    summary = {
        "name": name,
        "best_score": float(grid.best_score_),
        "best_params": grid.best_params_,
        "n_combos": n_combos,
        "n_fits": n_fits,
        "time_s": round(elapsed, 2),
    }
    print(
        f"\n[{name}]\tBest MCC = {grid.best_score_:.4f}\t\t|\t\t{n_combos} kombinacji × "
        f"{cv.get_n_splits()} foldów = {n_fits} fitów\t\t|\t\tczas = {elapsed:.1f}s"
    )
    print(f"[{name}]\tBest_params: {grid.best_params_}")
    return grid, summary
