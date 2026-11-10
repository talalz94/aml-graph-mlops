import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aml.cli
from aml import data as datamod
from aml.cli import _fresh, build_parser, main
from aml.evaluate import (
    best_f1_threshold,
    cut_for_recall,
    evaluate,
    recall_at_budget,
    typology_recall,
)
from aml.models import ALERT_SCORE, RULES, default_rules, rule_values, train_lightgbm, tune_rules
from aml.report import StaleResults, markers, render, render_baselines


def splits(frame):
    return [frame[frame["split"] == s].reset_index(drop=True) for s in ("train", "dev", "test")]


def wire_rows(n, **columns):
    """n Wire payments on which no rule fires, with the given columns set instead."""
    quiet = {
        "payment_format": "Wire",
        "is_laundering": 0,
        "self_loop": 0,
        "src_out_peers_day": 1,
        "dst_in_peers_day": 1,
        "reverse_payment_2d": 0,
        "pair_cnt_day": 1,
        "src_out_sum_day": 0.0,
        "src_in_sum_day": 0.0,
        "dst_out_sum_day": 0.0,
        "dst_in_sum_day": 0.0,
    }
    return pd.DataFrame(quiet | columns, index=range(n))


def test_tuned_rules_have_a_threshold_per_segment_and_rule(frame):
    train, _, _ = splits(frame)
    tuned = tune_rules(train)
    assert set(tuned.thresholds) == set(train["payment_format"].unique())
    assert all(set(t) == {rule for rule, _, _ in RULES} for t in tuned.thresholds.values())
    assert default_rules().alerts(train).dtype == bool


def test_tuned_rules_keep_the_catch_and_cut_the_noise(frame):
    """Per segment and rule: no more train alerts than the default threshold, and at least
    90% of the train laundering the default threshold catches."""
    train, _, _ = splits(frame)
    tuned = tune_rules(train)
    cut = 0
    for segment, part in train.groupby("payment_format"):
        ys, moved = part["is_laundering"].to_numpy(), part["self_loop"].to_numpy() == 0
        for rule, value, default in RULES:
            values = rule_values(part, value)
            base = (values >= default) & moved
            hit = (values >= tuned.threshold(segment, rule, default)) & moved
            assert hit.sum() <= base.sum(), (segment, rule)
            assert ys[hit].sum() >= 0.9 * ys[base].sum(), (segment, rule)
            cut += int(hit.sum() < base.sum())
    assert cut > 0  # tuning did remove alerts somewhere


def test_tuned_rule_never_fires_on_every_row_of_a_segment():
    # 6 laundering rows, none with a reverse payment or a repeated pair.
    n = 200
    part = wire_rows(
        n,
        is_laundering=[1] * 6 + [0] * (n - 6),
        reverse_payment_2d=[0] * 6 + [1] * 3 + [0] * (n - 9),
    )
    tuned = tune_rules(part)
    for rule, value, default in RULES:
        assert not (rule_values(part, value) >= tuned.threshold("Wire", rule, default)).all(), rule


def test_tuning_finds_the_fewest_alerts_even_for_a_rare_rule():
    """In 10,000 rows the default fan out cut (5 payees) raises 10 alerts and catches the one
    laundering row, which paid 9 payees; a cut at 9 keeps that catch with 1 alert. The
    default repeat pair cut (3 payments) also alerts on a clean pair of 4; a cut at 6 keeps
    only the laundering pair. No laundering pays in, so fan in rises to its top value."""
    peers = np.array([1] * 9_990 + [5] * 9 + [9])
    pairs = np.ones(len(peers), dtype=int)
    pairs[0], pairs[-1] = 4, 6
    senders = np.ones(len(peers), dtype=int)
    senders[:3] = (6, 8, 8)
    part = wire_rows(
        len(peers),
        is_laundering=(peers == 9).astype(int),
        src_out_peers_day=peers,
        pair_cnt_day=pairs,
        dst_in_peers_day=senders,
    )
    tuned = tune_rules(part)
    assert tuned.threshold("Wire", "fan_out", 5.0) == 9.0
    assert tuned.threshold("Wire", "repeat_pair", 3.0) == 6.0
    assert tuned.threshold("Wire", "fan_in", 5.0) == 8.0
    assert tuned.threshold("Wire", "round_trip", 1.0) == 1.0  # nothing fires, nothing to cut


def test_rules_never_fire_on_self_transfers(frame):
    train, _, _ = splits(frame)
    own = frame[frame["self_loop"] == 1]
    assert len(own) > 0
    assert not default_rules().alerts(own).any()
    assert not tune_rules(train).alerts(own).any()


def test_pass_through_needs_money_in_and_out_to_match():
    rows = pd.DataFrame(
        {
            "payment_format": "ACH",
            "paid_usd": 100.0,
            "self_loop": 0,
            "src_out_peers_day": 1,
            "dst_in_peers_day": 1,
            "reverse_payment_2d": 0,
            "pair_cnt_day": 1,
            "src_out_sum_day": [10_000.0, 9_500.0, 480.0],
            "src_in_sum_day": [1.0, 10_000.0, 500.0],
            "dst_out_sum_day": 0.0,
            "dst_in_sum_day": 100.0,
        }
    )
    # a token inflow, a matched flow, and a matched flow below the floor
    assert default_rules().alerts(rows).tolist() == [False, True, False]


def test_lightgbm_beats_chance_and_stops_on_average_precision(frame):
    train, dev, test = splits(frame)
    system = train_lightgbm(train, dev)
    assert list(system.model.evals_result_["valid_0"]) == ["average_precision"]
    result = evaluate(dev, test, system.score(dev), system.score(test))
    assert result["threshold_from"] == "dev"
    assert result["pr_auc"] > test["is_laundering"].mean()
    budgets = result["recall_at_budget"]
    assert budgets["50"] <= budgets["100"] <= budgets["500"]


def test_rules_are_scored_at_their_own_alerts(frame):
    _, dev, test = splits(frame)
    system = default_rules()
    result = evaluate(dev, test, system.score(dev), system.score(test), threshold=ALERT_SCORE)
    assert result["threshold_from"] == "fixed"
    assert result["alerts"] == int(system.alerts(test).sum())


def test_metric_helpers():
    y = np.array([0, 1, 0, 1, 0])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    assert cut_for_recall(y, score, 1.0) == 0.8
    assert cut_for_recall(y, score, 0.5) == 0.9
    # the lowest score that flags both laundering rows and nothing else
    assert best_f1_threshold(y, score) == 0.8
    assert np.array_equal(score >= best_f1_threshold(y, score), y == 1)
    df = pd.DataFrame({"day": [1, 1, 1, 2, 2], "is_laundering": y})
    assert recall_at_budget(df, score, 1) == 1.0  # each day's laundering ranks first
    worse = score.copy()
    worse[4] = 0.95  # a clean payment now outranks day 2's laundering
    assert recall_at_budget(df, worse, 1) == 0.5


def test_a_rule_budget_holds_only_the_rules_own_alerts():
    """One rule alert on a clean row and a budget of 2: the laundering row no rule fired on
    must not count as caught just because the budget had room left."""
    df = pd.DataFrame({"day": [1, 1], "is_laundering": [0, 1], "txn_id": ["a", "b"]})
    score = np.array([ALERT_SCORE, 0.0])
    assert recall_at_budget(df, score, 2) == 1.0  # ranking every row would count it
    assert recall_at_budget(df, score, 2, min_score=ALERT_SCORE) == 0.0
    # evaluate() fills a rule system's budgets the same way
    rows = pd.DataFrame({"day": [1, 1], "is_laundering": [1, 0]})
    rules_score = np.array([0.0, 2.0])
    result = evaluate(rows, rows, rules_score, rules_score, threshold=ALERT_SCORE)
    assert result["recall_at_budget"] == {"50": 0.0, "100": 0.0, "500": 0.0}


def test_typology_recall_counts_only_laundering_rows():
    key = {
        "timestamp": "2022/09/01 10:00",
        "from_bank": "001",
        "from_account": "A",
        "to_bank": "002",
        "to_account": "B",
        "amount_paid": "10.00",
    }
    rows = pd.DataFrame(
        [
            key | {"txn_id": "a", "day": 1, "is_laundering": 1},
            key | {"txn_id": "b", "day": 1, "is_laundering": 0},
        ]
    )
    patterns = pd.DataFrame([key | {"pattern": "CYCLE"}])
    out = typology_recall(rows, np.array([0.9, 0.1]), 1, patterns)
    assert out == {"CYCLE": {"n": 1, "recall": 1.0}}
    # with only the clean row alerted, spare budget does not reach the laundering row
    no_alert = np.array([0.0, ALERT_SCORE])
    assert typology_recall(rows, no_alert, 2, patterns)["CYCLE"]["recall"] == 1.0
    out = typology_recall(rows, no_alert, 2, patterns, min_score=ALERT_SCORE)
    assert out == {"CYCLE": {"n": 1, "recall": 0.0}}


def test_results_table_shows_recall_next_to_alerts():
    row = {
        "label": "LightGBM on graph features",
        "minority_f1": 0.5,
        "pr_auc": 0.4,
        "recall_at_budget": {"50": 0.1, "100": 0.2, "500": 0.3},
        "alerts_at_rules_recall": {
            "alerts": 1200,
            "recall": 0.41,
            "rules_alerts": 5000,
            "rules_recall": 0.4,
        },
        "command": "uv run aml eval --system lightgbm --dataset HI-Small",
        "test_transactions": 1000,
        "test_laundering": 10,
    }
    table = render({("HI-Small", "lightgbm"): row})
    assert "| Alerts (recall) vs tuned rules |" in table
    assert "| 1,200 (41.0%) vs 5,000 (40.0%) |" in table


def test_readme_baselines_are_rendered_from_report():
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    start, end = markers("baselines")
    block = readme[readme.index(start) + len(start) + 1 : readme.index(end)]
    assert block == render_baselines()
    assert "| GIN | 28.70 |" in block


def test_each_command_takes_only_the_options_it_uses():
    parser = build_parser()
    with pytest.raises(SystemExit):  # eval never touches the README
        parser.parse_args(["eval", "--system", "lightgbm", "--readme", "README.md"])
    report = parser.parse_args(["report"])
    assert (report.results_dir, report.readme) == (None, Path("README.md"))
    assert aml.cli._results_dir(report) == Path("results")  # report reads saved scores there


def test_reproduce_on_generated_data(tmp_path, syn_files, monkeypatch):
    data_dir = tmp_path / "data"
    raw = data_dir / "raw"
    raw.mkdir(parents=True)
    for path in syn_files:
        (raw / path.name).write_bytes(path.read_bytes())
    readme = tmp_path / "README.md"
    readme.write_text(
        "".join(f"{start}\n{end}\n" for start, end in map(markers, ("results", "baselines"))),
        encoding="utf-8",
    )
    common = ["--results-dir", str(tmp_path / "out"), "--readme", str(readme)]
    assert main(["--data-dir", str(data_dir), "reproduce", "--dataset", "SYN", *common]) == 0

    result = json.loads((tmp_path / "out" / "SYN" / "lightgbm.json").read_text(encoding="utf-8"))
    typology = result["typology_recall_at_100"]
    assert typology and set(typology) <= {"FAN-OUT", "FAN-IN", "CYCLE", "SCATTER-GATHER"}
    # every SYN laundering row is planted, so each one must match exactly one pattern row
    assert sum(v["n"] for v in typology.values()) == result["test_laundering"]
    assert result["best_iteration"] > 0
    # each result names the files and the code it came from
    assert result["inputs"]["trans_sha256"] == datamod.sha256(raw / "SYN_Trans.csv")
    assert result["inputs"]["patterns_sha256"] == datamod.sha256(raw / "SYN_Patterns.txt")
    # the tuned rules row is the bar itself
    tuned = json.loads((tmp_path / "out" / "SYN" / "rules-tuned.json").read_text(encoding="utf-8"))
    bar = tuned["alerts_at_rules_recall"]
    assert (bar["alerts"], bar["recall"]) == (bar["rules_alerts"], bar["rules_recall"])
    assert tuned["alerts"] == bar["rules_alerts"]
    # a rule system's budgets hold only its own alerts, so no budget beats its own recall
    for name in ("rules-default", "rules-tuned"):
        saved = json.loads((tmp_path / "out" / "SYN" / f"{name}.json").read_text(encoding="utf-8"))
        assert max(saved["recall_at_budget"].values()) <= saved["recall"] + 1e-12, name

    assert main(["--data-dir", str(data_dir), "report", *common]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "No results yet" in text  # generated data is never shown
    assert "| GIN | 28.70 |" in text

    # eval reuses a fresh table, and without --results-dir generated scores stay in the
    # data folder (the working directory is moved so nothing can reach the repo's results/)
    monkeypatch.chdir(tmp_path)
    table = data_dir / "features" / "SYN.parquet"
    built = table.stat().st_mtime_ns
    one = ["--data-dir", str(data_dir), "eval", "--system", "rules-default", "--dataset", "SYN"]
    assert main(one) == 0
    assert table.stat().st_mtime_ns == built
    assert (data_dir / "results" / "SYN" / "rules-default.json").exists()
    assert not (tmp_path / "results").exists()

    # reproduce rebuilds even a fresh table
    calls = []
    monkeypatch.setattr(aml.cli, "cmd_features", calls.append)
    monkeypatch.setattr(aml.cli, "SYSTEMS", ("rules-default",))
    assert main(["--data-dir", str(data_dir), "reproduce", "--dataset", "SYN", *common]) == 0
    assert len(calls) == 1

    # a changed raw file makes the saved table stale
    args = build_parser().parse_args(one)
    assert _fresh(args)
    with (raw / "SYN_Trans.csv").open("a", encoding="utf-8", newline="") as handle:
        handle.write("2022/09/10 23:59,001,X,002,Y,1.00,US Dollar,1.00,US Dollar,ACH,0\n")
    assert not _fresh(args)


def test_tuning_keeps_the_default_when_a_higher_cut_removes_no_alert():
    # No laundering, and all seven rows at or above the default fan in cut have 7 senders:
    # a cut at 7 raises the same seven train alerts, so the default stays.
    n = 100
    part = wire_rows(n, dst_in_peers_day=[7] * 7 + [1] * (n - 7))
    assert tune_rules(part).threshold("Wire", "fan_in", 5.0) == 5.0


def test_report_refuses_results_from_different_builds():
    def row(system, package):
        return {
            "label": system,
            "minority_f1": 0.5,
            "pr_auc": 0.4,
            "recall_at_budget": {"50": 0.1, "100": 0.2, "500": 0.3},
            "alerts_at_rules_recall": {
                "alerts": 1,
                "recall": 0.1,
                "rules_alerts": 1,
                "rules_recall": 0.1,
            },
            "command": "x",
            "test_transactions": 1000,
            "test_laundering": 10,
            "created_utc": package,
            "inputs": {
                "trans_sha256": "t",
                "period_end": "p",
                "code_sha256": "c",
                "package_sha256": package,
            },
        }

    same = {
        ("HI-Small", "lightgbm"): row("lightgbm", "a"),
        ("HI-Small", "rules-tuned"): row("rules-tuned", "a"),
    }
    render(same)
    mixed = {
        ("HI-Small", "lightgbm"): row("lightgbm", "b"),
        ("HI-Small", "rules-tuned"): row("rules-tuned", "a"),
    }
    with pytest.raises(StaleResults, match="rules-tuned"):
        render(mixed)


def test_a_patterns_file_with_no_transactions_is_skipped(tmp_path):
    import argparse

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "SYN_Patterns.txt").write_text("", encoding="utf-8")
    args = argparse.Namespace(data_dir=tmp_path, dataset="SYN")
    assert aml.cli._patterns(args) is None
