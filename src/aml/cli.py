"""Command line entry point.

aml download [--dataset HI-Small]   fetch from Kaggle (credentials from the environment)
aml import FILE [--dataset X]       use a file, or a zip, you downloaded from the Kaggle page
aml synth                           write a small generated dataset (SYN) for tests and the demo
aml features [--dataset X]          build the feature table with PySpark
aml eval --system NAME [--dataset X]
                                    one system; rebuilds the features if they are out of date
aml reproduce [--dataset X]         rebuild the features and every system, then the README
                                    tables (not for SYN)
aml report                          rewrite the README's generated blocks from saved files

Scores on generated data (SYN) go to DATA_DIR/results unless --results-dir says otherwise;
aml report never renders them and git ignores results/SYN/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from aml import data as datamod
from aml.evaluate import cut_for_recall, evaluate, typology_recall
from aml.models import ALERT_SCORE, RuleSystem, default_rules, train_lightgbm, tune_rules
from aml.report import (
    StaleResults,
    load_results,
    render,
    render_baselines,
    save_result,
    update_readme,
)
from aml.synth import generate

SYSTEMS = ("rules-default", "rules-tuned", "lightgbm")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _paths(args) -> tuple[Path, Path, Path]:
    raw = args.data_dir / "raw"
    return (
        raw / f"{args.dataset}_Trans.csv",
        raw / f"{args.dataset}_Patterns.txt",
        args.data_dir / "features" / f"{args.dataset}.parquet",
    )


def _period_end(dataset: str) -> str:
    if dataset == "SYN":  # synth writes Sep 1 to 10, the same window as Small
        return datamod.PERIOD_END["Small"]
    return datamod.PERIOD_END[datamod.size_of(dataset)]


def _results_dir(args) -> Path:
    """--results-dir when given. Otherwise results/, except that scores on generated data go
    under the data folder, which git ignores."""
    if args.results_dir is not None:
        return args.results_dir
    if getattr(args, "dataset", None) == "SYN":
        return args.data_dir / "results"
    return Path("results")


def _generated(args) -> str:
    return " (generated data)" if args.dataset == "SYN" else ""


def cmd_download(args) -> int:
    print(datamod.NOTICE)
    for path, digest in datamod.download(args.data_dir, args.dataset):
        print(f"  {path.as_posix()}  sha256 {digest}")
    return 0


def cmd_import(args) -> int:
    for path, digest in datamod.import_file(args.data_dir, args.file, args.dataset):
        print(f"Imported {path.as_posix()}  sha256 {digest}")
    return 0


def cmd_synth(args) -> int:
    trans, patterns = generate(args.data_dir / "raw", prefix="SYN", seed=args.seed)
    print(
        f"Wrote {trans.as_posix()} and {patterns.as_posix()} "
        "(generated data for tests and the demo; never quote numbers from it)"
    )
    return 0


def _code_sha256() -> str:
    """Hash of the code that shapes the feature table, so a change to it forces a rebuild."""
    from aml import features, schema

    digest = hashlib.sha256()
    for module in (features, schema):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def _package_sha256() -> str:
    """Hash of every module in the package (rules, model, metrics and pattern reader as well
    as the features), so a result names the code that scored it."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _stamp(args) -> dict:
    """What a feature table is built from. It is saved next to the table, and every result
    carries it together with a hash of the whole package (see _evaluate)."""
    trans, _, _ = _paths(args)
    return {
        "trans_file": trans.name,
        "trans_sha256": datamod.sha256(trans),
        "period_end": _period_end(args.dataset),
        "code_sha256": _code_sha256(),
    }


def cmd_features(args) -> int:
    from aml.features import build_frame, save

    trans, _, out = _paths(args)
    if not trans.exists():
        get = "`aml synth`" if args.dataset == "SYN" else "`aml download` or `aml import`"
        raise FileNotFoundError(f"{trans} not found. Run {get} first.")
    stamp_path = out.with_suffix(".json")
    stamp_path.unlink(missing_ok=True)  # a stamp only ever describes a finished table
    _log(f"Building features for {args.dataset} with PySpark ...")
    stamp = _stamp(args)
    frame = build_frame(trans, period_end=stamp["period_end"])
    save(frame, out)
    stamp["built_utc"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    counts = frame.groupby("split")["is_laundering"].agg(["size", "sum"]).to_dict("index")
    _log(f"Saved {out.as_posix()}: {json.dumps(counts)}")
    return 0


def _fresh(args) -> bool:
    """True when the saved feature table still matches its stamp: the same raw file (when
    it is on disk to compare), the same period end and the same feature code."""
    trans, _, table = _paths(args)
    stamp_path = table.with_suffix(".json")
    if not (table.exists() and stamp_path.exists()):
        return False
    saved = json.loads(stamp_path.read_text(encoding="utf-8"))
    current = {"period_end": _period_end(args.dataset), "code_sha256": _code_sha256()}
    if trans.exists():
        current["trans_sha256"] = datamod.sha256(trans)
    return all(saved.get(k) == v for k, v in current.items())


def _frame(args, *, rebuild: bool = False) -> tuple[pd.DataFrame, dict]:
    """The feature table and its stamp. `aml eval` reuses a saved table while it is fresh;
    `aml reproduce` always rebuilds it."""
    if rebuild or not _fresh(args):
        cmd_features(args)
    _, _, table = _paths(args)
    stamp = json.loads(table.with_suffix(".json").read_text(encoding="utf-8"))
    return pd.read_parquet(table), stamp


PATTERN_KEY = ["timestamp", "from_bank", "from_account", "to_bank", "to_account", "amount_paid"]


def _patterns(args) -> pd.DataFrame | None:
    _, path, _ = _paths(args)
    if not path.exists():
        return None
    found = datamod.read_patterns(path)
    if not found:
        _log(
            f"warning: {path.name} lists no laundering transactions, "
            "so per pattern recall is skipped"
        )
        return None
    rows = [dict(zip(PATTERN_KEY, p.key, strict=True)) | {"pattern": p.kind} for p in found]
    return pd.DataFrame(rows).drop_duplicates(subset=PATTERN_KEY)


def _evaluate(name: str, args, frame: pd.DataFrame, stamp: dict) -> dict:
    train, dev, test = (
        frame[frame["split"] == s].reset_index(drop=True) for s in ("train", "dev", "test")
    )
    if name == "rules-default":
        system = default_rules()
    elif name == "rules-tuned":
        system = tune_rules(train)
    else:
        system = train_lightgbm(train, dev)
    is_rules = isinstance(system, RuleSystem)
    dev_score, test_score = system.score(dev), system.score(test)
    # The rules are scored at their own alerts, which are also all that fill their daily
    # budgets; LightGBM at the threshold with the best F1 on dev, ranking every row.
    own_cut = ALERT_SCORE if is_rules else None
    result = evaluate(dev, test, dev_score, test_score, threshold=own_cut)
    # Test alerts and recall beside the tuned rules' own: a rule system at its own alerts,
    # LightGBM at a cut set on dev to reach the tuned rules' recall there.
    rules = system if name == "rules-tuned" else tune_rules(train)
    y_dev, y = dev["is_laundering"].to_numpy(), test["is_laundering"].to_numpy()
    rule_alerts = rules.alerts(test)
    if is_rules:
        own = system.alerts(test)  # a rule system's real operating point: any rule fired
    else:
        dev_recall = float(rules.alerts(dev)[y_dev == 1].mean()) if y_dev.any() else 0.0
        own = test_score >= cut_for_recall(y_dev, dev_score, dev_recall)
    result["alerts_at_rules_recall"] = {
        "rules_recall": float(rule_alerts[y == 1].mean()) if y.any() else 0.0,
        "rules_alerts": int(rule_alerts.sum()),
        "alerts": int(own.sum()),
        "recall": float(own[y == 1].mean()) if y.any() else 0.0,
    }
    inputs = dict(stamp) | {"package_sha256": _package_sha256()}
    patterns = _patterns(args)
    if patterns is not None:
        typology = typology_recall(test, test_score, 100, patterns, min_score=own_cut)
        if not typology and y.any():
            _log("warning: no test laundering matched the patterns file, check the key columns")
        result["typology_recall_at_100"] = typology
        patterns_path = _paths(args)[1]
        inputs["patterns_file"] = patterns_path.name
        inputs["patterns_sha256"] = datamod.sha256(patterns_path)
    if name == "rules-tuned":
        result["thresholds"] = system.thresholds
    if name == "lightgbm":
        result["best_iteration"] = int(system.model.best_iteration_)
    result["inputs"] = inputs
    return result


def cmd_eval(args) -> int:
    frame, stamp = _frame(args)
    result = _evaluate(args.system, args, frame, stamp)
    path = save_result(_results_dir(args), args.dataset, args.system, result)
    f1, auc = 100 * result["minority_f1"], 100 * result["pr_auc"]
    print(
        f"{args.system}: minority F1 {f1:.1f}, PR-AUC {auc:.1f}{_generated(args)}  "
        f"saved {path.as_posix()}"
    )
    return 0


def cmd_reproduce(args) -> int:
    frame, stamp = _frame(args, rebuild=True)  # always rebuilt, so results match the code
    for name in SYSTEMS:
        args.system = name
        result = _evaluate(name, args, frame, stamp)
        path = save_result(_results_dir(args), args.dataset, name, result)
        f1 = 100 * result["minority_f1"]
        _log(f"{name}: minority F1 {f1:.1f}{_generated(args)}  saved {path.as_posix()}")
    return cmd_report(args) if args.dataset != "SYN" else 0


def cmd_report(args) -> int:
    results = load_results(_results_dir(args))
    blocks = {"results": render(results), "baselines": render_baselines()}
    changed = update_readme(args.readme, blocks)
    print(f"{args.readme} {'updated' if changed else 'already up to date'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aml", description="AML transaction monitoring.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)

    def dataset(p, choices=(*datamod.DATASETS, "SYN"), text=None):
        p.add_argument("--dataset", default="HI-Small", choices=list(choices), help=text)

    def results_dir(p, text="where scores go (default: results, or DATA_DIR/results for SYN)"):
        p.add_argument("--results-dir", type=Path, default=None, help=text)

    def readme(p):
        p.add_argument(
            "--readme",
            type=Path,
            default=Path("README.md"),
            help="the README whose tables are rewritten (default: README.md)",
        )

    p = sub.add_parser("download", help="fetch from Kaggle with credentials from the environment")
    dataset(p, choices=datamod.DATASETS)  # generated data is never downloaded
    p.set_defaults(func=cmd_download)
    p = sub.add_parser("import", help="use a file, or a zip, you downloaded yourself")
    p.add_argument("file", type=Path)
    dataset(p, text="which dataset's files to take from a zip")
    p.set_defaults(func=cmd_import)
    p = sub.add_parser("synth", help="write a small generated dataset for tests and the demo")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_synth)
    p = sub.add_parser("features", help="build the feature table with PySpark")
    dataset(p)
    p.set_defaults(func=cmd_features)
    p = sub.add_parser("eval", help="evaluate one system")
    dataset(p)
    p.add_argument("--system", choices=SYSTEMS, required=True)
    results_dir(p)
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser(
        "reproduce",
        help="rebuild the features, evaluate every system, rewrite the README tables (not for SYN)",
    )
    dataset(p)
    results_dir(p)
    readme(p)
    p.set_defaults(func=cmd_reproduce)
    p = sub.add_parser("report", help="rewrite the README's generated blocks")
    results_dir(p, "where saved scores are read from (default: results)")
    readme(p)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, PermissionError, datamod.DataFileError, StaleResults) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except httpx.HTTPError as err:
        # The first line only, with any query string cut, so a signed link stays private.
        line = (str(err).splitlines() or [type(err).__name__])[0]
        line = re.sub(r"\?[^\s'\"]*", "", line)
        print(f"error: download failed: {line}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
