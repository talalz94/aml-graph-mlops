"""Detection systems: transaction monitoring rules (default and tuned) and LightGBM.

Rules are the classic scenarios a monitoring system ships with. "Default" uses fixed
thresholds. "Tuned" sets each threshold per payment format segment on the train period,
the way scenario tuning works at a bank: keep the catch, cut the noise. For each segment
and rule it tries the default threshold and every train value above it, and takes the one
that raises the fewest train alerts while still catching at least 90% of the train
laundering the default threshold catches. The default itself always qualifies, so a tuned
rule alerts on a subset of what its default alerts on.

A self transfer moves no money between accounts, so no rule ever fires on one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aml.features import CATEGORICAL, FEATURES

# (name, value, default threshold). A rule fires when value >= threshold.
RULES = [
    ("fan_out", "src_out_peers_day", 5.0),
    ("fan_in", "dst_in_peers_day", 5.0),
    ("pass_through_sender", "src_flow_balance_day", 0.9),
    ("pass_through_receiver", "dst_flow_balance_day", 0.9),
    ("round_trip", "reverse_payment_2d", 1.0),
    ("repeat_pair", "pair_cnt_day", 3.0),
]
SEGMENT = "payment_format"
ALERT_SCORE = 1.0  # a transaction is alerted when at least one rule fires
MIN_PASS_IN = 1_000.0  # US Dollar coming in that day before a pass through rule can fire
# Rule values worked out from the day's money out and in, rather than read from a column.
FLOW_BALANCE = {
    "src_flow_balance_day": ("src_out_sum_day", "src_in_sum_day"),
    "dst_flow_balance_day": ("dst_out_sum_day", "dst_in_sum_day"),
}


def _flow_balance(out_sum: pd.Series, in_sum: pd.Series) -> np.ndarray:
    """The smaller of the day's money in and out over the larger: 1.0 when out equals in,
    at least 0.9 when out is 90% to 111% of in, near 0 when either side dominates, and 0
    when less than MIN_PASS_IN came in."""
    o, i = out_sum.to_numpy(dtype=float), in_sum.to_numpy(dtype=float)
    big = np.maximum(o, i)
    share = np.divide(np.minimum(o, i), big, out=np.zeros_like(big), where=big > 0)
    return np.where(i >= MIN_PASS_IN, share, 0.0)


def rule_values(df: pd.DataFrame, value: str) -> np.ndarray:
    """What a rule compares with its threshold: a feature column, or for the pass through
    rules the two sided flow balance of the day's money out and in."""
    if value in FLOW_BALANCE:
        out_sum, in_sum = FLOW_BALANCE[value]
        return _flow_balance(df[out_sum], df[in_sum])
    return df[value].to_numpy()


@dataclass
class RuleSystem:
    """Score = number of rules fired, with the US Dollar amount as a small tie breaker, so
    alerts can also be ranked when a budget is smaller than the number of rule hits."""

    name: str
    thresholds: dict[str, dict[str, float]] = field(default_factory=dict)  # segment -> rule -> t

    def threshold(self, segment: str, rule: str, default: float) -> float:
        return self.thresholds.get(segment, {}).get(rule, default)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        fired = np.zeros(len(df), dtype=np.float64)
        segments = df[SEGMENT].to_numpy()
        for rule, value, default in RULES:
            t = np.array([self.threshold(s, rule, default) for s in segments])
            fired += (rule_values(df, value) >= t).astype(np.float64)
        # A transfer from an account to itself is not a round trip or a pass through.
        fired[df["self_loop"].to_numpy() == 1] = 0.0
        # Well under 1 for any real amount, so it never lifts a row past a rule count.
        tie = np.log1p(df["paid_usd"].to_numpy()) / 100.0
        return fired + np.where(fired > 0, tie, 0.0)

    def alerts(self, df: pd.DataFrame) -> np.ndarray:
        return self.score(df) >= ALERT_SCORE


def default_rules() -> RuleSystem:
    return RuleSystem("rules-default")


def tune_rules(train: pd.DataFrame, *, keep: float = 0.9) -> RuleSystem:
    """Per payment format segment and rule, try the default threshold and every train value
    above it, and pick the one that raises the fewest train alerts while still catching at
    least `keep` of the train laundering that the default threshold catches. The default
    always qualifies, so every segment and rule gets a threshold, and a lower cut, such as
    one that fires on every row, can never win. A higher cut wins only when it raises fewer
    train alerts than the default, so where the default catches no train laundering in a
    segment, the threshold rises to the segment's highest train value, unless every row at
    or above the default shares that value; then the default stays."""
    system = RuleSystem("rules-tuned")
    for segment, part in train.groupby(SEGMENT):
        moved = part["self_loop"].to_numpy() == 0  # rules never fire on self transfers
        ys = part["is_laundering"].to_numpy()[moved]
        for rule, value, default in RULES:
            values = rule_values(part, value)[moved]
            at_default = values >= default
            need = keep * int(ys[at_default].sum())
            best_t, best_alerts = default, int(at_default.sum())
            if best_alerts:
                # Sorted from the highest value down, a cut at ranked[k], where k is the last
                # place holding that value, alerts on exactly the first k + 1 rows.
                order = np.argsort(values, kind="stable")[::-1]
                ranked, caught = values[order], np.cumsum(ys[order])
                last = np.r_[np.nonzero(np.diff(ranked))[0], len(ranked) - 1]
                ok = (caught[last] >= need) & (last + 1 < best_alerts)
                if ok.any():  # the first cut that qualifies is the highest: fewest alerts
                    k = last[np.argmax(ok)]
                    best_t, best_alerts = float(ranked[k]), int(k + 1)
            system.thresholds.setdefault(str(segment), {})[rule] = best_t
    return system


@dataclass
class GBMSystem:
    name: str
    model: object
    features: list[str]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features])[:, 1]


def train_lightgbm(train: pd.DataFrame, dev: pd.DataFrame, *, seed: int = 7) -> GBMSystem:
    import lightgbm as lgb

    pos = max(int(train["is_laundering"].sum()), 1)
    neg = len(train) - pos
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=min(neg / pos, 50.0),
        metric="average_precision",  # the only dev metric, so early stopping watches AP alone
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(
        train[FEATURES],
        train["is_laundering"],
        eval_X=(dev[FEATURES],),
        eval_y=(dev["is_laundering"],),
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)],
        categorical_feature=CATEGORICAL,
    )
    return GBMSystem("lightgbm", model, list(FEATURES))
