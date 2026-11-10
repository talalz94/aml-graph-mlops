"""Metrics an AML team and a benchmark reader both care about.

minority F1         the illicit class F1, the paper's metric. LightGBM at the threshold
                    with the best F1 on dev (the GNN code released with the paper takes
                    the argmax of its output instead); the rules at their own alerts,
                    whenever any rule fires
PR-AUC              average precision on test, threshold free
recall at budget    laundering caught when only the top K alerts a day can be reviewed;
                    for the rules only their own alerts count, so on a day with fewer
                    than K of them the rest of the budget goes unused
alerts at the tuned rules' recall
                    each system's test alerts and recall beside the tuned rules' own: the
                    rules at their own alerts, LightGBM at a cut set on dev to reach the
                    tuned rules' recall on dev (cut_for_recall)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

BUDGETS = (50, 100, 500)  # alerts reviewed per day


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))]) if len(thresholds) else 0.5


def _reviewed(
    df: pd.DataFrame, score: np.ndarray, per_day: int, min_score: float | None
) -> pd.DataFrame:
    """Each day's top `per_day` rows by score. With `min_score`, only rows scoring at least
    that are alerts, so a day with fewer alerts than the budget leaves the rest unused."""
    ranked = df.assign(_s=score)
    if min_score is not None:
        ranked = ranked[ranked["_s"] >= min_score]
    ranked = ranked.sort_values(["day", "_s"], ascending=[True, False], kind="stable")
    return ranked.groupby("day", sort=False).head(per_day)


def recall_at_budget(
    df: pd.DataFrame, score: np.ndarray, per_day: int, *, min_score: float | None = None
) -> float:
    """Share of all the laundering in `df` that the reviewed alerts catch (see _reviewed)."""
    top = _reviewed(df, score, per_day, min_score)
    total = int(df["is_laundering"].sum())
    return float(top["is_laundering"].sum() / total) if total else float("nan")


def cut_for_recall(y: np.ndarray, score: np.ndarray, recall: float) -> float:
    """Highest score cut that reaches `recall` on these labels. Pick it on dev, apply to test."""
    order = np.argsort(-score, kind="stable")
    hit = np.nonzero(np.cumsum(y[order]) >= recall * y.sum() - 1e-9)[0]
    return float(score[order][hit[0]]) if len(hit) else float(score.min())


def evaluate(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    dev_score: np.ndarray,
    test_score: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict:
    """Test metrics. The F1 threshold is the best one on dev unless `threshold` fixes it,
    as for the rules, whose own alerts are the operating point; then only those alerts
    fill the daily budgets as well."""
    y_dev, y_test = dev["is_laundering"].to_numpy(), test["is_laundering"].to_numpy()
    t = best_f1_threshold(y_dev, dev_score) if threshold is None else threshold
    pred = test_score >= t
    return {
        "threshold": t,
        "threshold_from": "dev" if threshold is None else "fixed",
        "minority_f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(y_test[pred].mean()) if pred.any() else 0.0,
        "recall": float(pred[y_test == 1].mean()) if y_test.any() else float("nan"),
        "pr_auc": float(average_precision_score(y_test, test_score))
        if y_test.any()
        else float("nan"),
        "alerts": int(pred.sum()),
        "recall_at_budget": {
            str(k): recall_at_budget(test, test_score, k, min_score=threshold) for k in BUDGETS
        },
        "test_transactions": len(test),
        "test_laundering": int(y_test.sum()),
    }


def typology_recall(
    test: pd.DataFrame,
    score: np.ndarray,
    per_day: int,
    patterns: pd.DataFrame,
    *,
    min_score: float | None = None,
) -> dict:
    """Recall by laundering pattern at a daily budget, with the budget filled as in
    recall_at_budget. `patterns` maps txn keys to types; only laundering rows are matched,
    so a clean row that shares a key is never counted."""
    flagged = set(_reviewed(test, score, per_day, min_score)["txn_id"])
    joined = test[test["is_laundering"] == 1].merge(
        patterns,
        on=["timestamp", "from_bank", "from_account", "to_bank", "to_account", "amount_paid"],
        validate="many_to_one",
    )
    out = {}
    for kind, part in joined.groupby("pattern"):
        out[kind] = {"n": len(part), "recall": float(part["txn_id"].isin(flagged).mean())}
    return out
