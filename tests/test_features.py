import csv
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from aml.data import PERIOD_END, read_patterns
from aml.features import FEATURES, FILLED, HISTORY, build, temporal_split, to_pandas
from aml.models import default_rules
from aml.schema import RAW_HEADER
from aml.synth import USD_RATES

# Raw columns that identify a row without its label (txn_id hashes the label too).
RAW_KEY = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
]
# Everything a later day or a label could leak into: the features and the US Dollar amounts.
CHECKED = [*FEATURES, "paid_usd", "received_usd"]


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        header, *rows = csv.reader(handle)
    return header, rows


def write_rows(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return path


@pytest.fixture(scope="module")
def variants(spark, syn_files, tmp_path_factory):
    """The generated file cut after each of its days, and the file with its labels
    shuffled across rows. Each build is mostly Spark's fixed overhead, so a few run at
    once in the shared session."""
    header, rows = read_rows(syn_files[0])
    folder = tmp_path_factory.mktemp("variants")
    days = sorted({r[0][:10] for r in rows})
    paths = {
        day: write_rows(
            folder / f"cut_{day.replace('/', '')}.csv",
            header,
            [r for r in rows if r[0][:10] <= day],
        )
        for day in days[:-1]
    }
    labels = [r[-1] for r in rows]
    random.Random(5).shuffle(labels)
    paths["shuffled"] = write_rows(
        folder / "shuffled.csv", header, [[*r[:-1], y] for r, y in zip(rows, labels, strict=True)]
    )

    def run(path):
        return to_pandas(build(spark, path, period_end=PERIOD_END["Small"]))

    with ThreadPoolExecutor(3) as pool:
        return dict(zip(paths, pool.map(run, paths.values()), strict=True))


def test_every_feature_is_present_and_filled(frame):
    assert set(FEATURES) <= set(frame.columns)
    assert not frame[FILLED].isna().any().any()
    first = frame["day"] == frame["day"].min()
    assert frame.loc[first, HISTORY].isna().all().all()  # no days before the first
    assert not frame.loc[~first, HISTORY].isna().any().any()
    assert frame["txn_id"].is_unique


def test_split_is_temporal_60_20_20_in_whole_days(frame):
    shares = frame["split"].value_counts(normalize=True)
    one_day = frame.groupby("day").size().max() / len(frame)
    for name, target in (("train", 0.6), ("dev", 0.2), ("test", 0.2)):
        assert abs(shares[name] - target) <= one_day
    day = {s: frame.loc[frame["split"] == s, "day"] for s in ("train", "dev", "test")}
    assert day["train"].max() < day["dev"].min()
    assert day["dev"].max() < day["test"].min()


def test_temporal_split_picks_the_closest_whole_days():
    pdf = pd.DataFrame({"day": np.repeat([1, 2, 3, 4, 5, 6], [50, 10, 10, 10, 10, 10])})
    split = temporal_split(pdf).groupby(pdf["day"])
    assert split.nunique().max() == 1
    assert split.first().tolist() == ["train", "train", "dev", "dev", "test", "test"]  # 60/20/20
    with pytest.raises(ValueError, match="3 days"):
        temporal_split(pdf[pdf["day"] <= 2])


def test_planted_fan_patterns_show_up(frame, syn_files):
    kinds = {}
    for p in read_patterns(syn_files[1]):
        kinds.setdefault(p.kind, set()).add(p.key)
    key = list(zip(frame["timestamp"], frame["from_bank"], frame["from_account"], frame["to_bank"],
                   frame["to_account"], frame["amount_paid"], strict=True))  # fmt: skip
    pattern_of = np.array([next((k for k, keys in kinds.items() if x in keys), "") for x in key])
    assert frame.loc[pattern_of == "FAN-OUT", "src_out_peers_day"].min() >= 5
    assert frame.loc[pattern_of == "FAN-IN", "dst_in_peers_day"].min() >= 5


def test_features_never_use_later_days(frame, variants):
    """Cut the file after each of its days in turn: no earlier transaction may change."""
    full = frame.set_index("txn_id")
    cuts = {k: v for k, v in variants.items() if k != "shuffled"}
    assert len(cuts) == frame["day"].nunique() - 1
    for day, part in cuts.items():
        part = part.set_index("txn_id")
        assert 0 < len(part) < len(full), day
        left = full.loc[part.index, CHECKED].to_numpy(dtype=float)
        assert np.allclose(left, part[CHECKED].to_numpy(dtype=float), equal_nan=True), day


def test_features_never_use_the_label(frame, variants):
    """Shuffle the labels across rows: no feature, US Dollar amount or split may change."""
    other = variants["shuffled"]
    other = other.assign(split=temporal_split(other))
    left = frame.sort_values([*RAW_KEY, *CHECKED], kind="stable").reset_index(drop=True)
    right = other.sort_values([*RAW_KEY, *CHECKED], kind="stable").reset_index(drop=True)
    assert (left["is_laundering"] != right["is_laundering"]).any()  # the labels did move
    assert left["is_laundering"].sum() == right["is_laundering"].sum()
    assert np.allclose(
        left[CHECKED].to_numpy(dtype=float), right[CHECKED].to_numpy(dtype=float), equal_nan=True
    )
    assert (left["split"] == right["split"]).all()


def test_history_features_share_a_scale(frame):
    """A steady account looks the same on a train day and a test day."""
    later = frame["day"] > frame["day"].min()
    train = frame.loc[later & (frame["split"] == "train"), "src_out_cnt_7d_avg"].mean()
    test = frame.loc[frame["split"] == "test", "src_out_cnt_7d_avg"].mean()
    assert 0.7 < train / test < 1.3


def test_self_transfers_are_not_2_cycles_or_pairs(frame):
    own = frame[frame["self_loop"] == 1]
    assert len(own) > 0 and (own["reverse_payment_2d"] == 0).all()
    assert (own["pair_cnt_day"] == 0).all()
    assert (frame.loc[frame["self_loop"] == 0, "pair_cnt_day"] >= 1).all()  # counts itself


def test_amounts_are_in_us_dollar_at_the_generated_rates(frame):
    """From the first day a currency is paired with US Dollar, its amounts convert at the
    rate the generator used; before that day, at 1.0."""
    usd = "US Dollar"
    paired = frame[
        (frame["payment_currency"] != frame["receiving_currency"])
        & ((frame["payment_currency"] == usd) | (frame["receiving_currency"] == usd))
    ]
    other = paired["receiving_currency"].where(
        paired["payment_currency"] == usd, paired["payment_currency"]
    )
    first_seen = paired["day"].groupby(other).min().to_dict()
    assert set(first_seen) == set(USD_RATES) - {usd}
    converted = 0
    sides = (("amount_paid", "payment_currency", "paid_usd"),
             ("amount_received", "receiving_currency", "received_usd"))  # fmt: skip
    for amount, currency, in_usd in sides:
        raw = frame[amount].astype(float)
        rows = frame[raw > 0]
        pairs = zip(rows[currency], rows["day"], strict=True)
        known = np.array([c in first_seen and d >= first_seen[c] for c, d in pairs])
        expected = np.where(known, rows[currency].map(USD_RATES), 1.0)
        assert np.allclose(rows[in_usd] / raw[raw > 0], expected, rtol=0.02)
        converted += int((known & (rows[currency] != usd).to_numpy()).sum())
    assert converted > 1000


def toy_row(ts, src, dst, paid, currency="US Dollar", received=None, to_currency=None, fmt="ACH"):
    received, to_currency = received or paid, to_currency or currency
    return [ts, "001", src, "001", dst, received, to_currency, paid, currency, fmt, "0"]


TOY_ROWS = [
    # A pays B, then makes only a self transfer the next day.
    toy_row("2022/09/01 10:00", "A", "B", "100.00"),
    toy_row("2022/09/02 10:00", "A", "A", "50.00", fmt="Reinvestment"),
    # D pays C back the next day; F pays E back four days later.
    toy_row("2022/09/01 11:00", "C", "D", "200.00"),
    toy_row("2022/09/02 11:00", "D", "C", "190.00"),
    toy_row("2022/09/01 12:00", "E", "F", "300.00"),
    toy_row("2022/09/05 12:00", "F", "E", "290.00"),
    # G pays H every day from 1 to 8 September.
    *[toy_row(f"2022/09/0{d} 13:00", "G", "H", "10.00") for d in range(1, 9)],
    # J pays K a week apart, across the night London clocks go back.
    toy_row("2022/10/24 09:00", "J", "K", "10.00"),
    toy_row("2022/10/31 09:00", "J", "K", "10.00"),
    # M receives 100,000 Yen as 670 US Dollar and sends the 670 on.
    toy_row("2022/09/03 14:00", "L", "M", "100000.00", "Yen", "670.00", "US Dollar"),
    toy_row("2022/09/03 15:00", "M", "N", "670.00"),
    # P pays Q 100 Euro a day. Euro first pairs with US Dollar on the 2nd (at 1.17), then
    # again on the 3rd (at 1.20).
    *[toy_row(f"2022/09/0{d} 17:00", "P", "Q", "100.00", "Euro") for d in (1, 2, 3)],
    toy_row("2022/09/02 16:00", "R", "S", "100.00", "Euro", "117.00", "US Dollar"),
    toy_row("2022/09/03 16:00", "S", "R", "120.00", "US Dollar", "100.00", "Euro"),
]


@pytest.fixture(scope="module")
def toy(spark, tmp_path_factory):
    """TOY_ROWS built in a session time zone with daylight saving time, since no feature
    may depend on the session time zone."""
    path = write_rows(tmp_path_factory.mktemp("toy") / "TOY_Trans.csv", RAW_HEADER, TOY_ROWS)
    spark.conf.set("spark.sql.session.timeZone", "Europe/London")
    try:
        return to_pandas(build(spark, path))
    finally:
        spark.conf.set("spark.sql.session.timeZone", "UTC")


def pick(toy, src, dst, ts):
    rows = toy[(toy["from_account"] == src) & (toy["to_account"] == dst) & (toy["timestamp"] == ts)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_self_transfer_moves_no_money(toy):
    own = pick(toy, "A", "A", "2022/09/02 10:00")
    assert own["self_loop"] == 1
    for col in [
        "src_out_cnt_day",
        "src_out_sum_day",
        "src_out_peers_day",
        "src_in_sum_day",
        "src_pass_through_day",
        "dst_in_peers_day",
        "dst_pass_through_day",
        "reverse_payment_2d",
        "pair_cnt_day",
    ]:
        assert own[col] == 0, col
    assert own["src_out_cnt_7d_avg"] == 1.0  # the payment to B the day before still counts
    assert not default_rules().alerts(toy[toy["self_loop"] == 1]).any()


def test_2_cycle_needs_the_payment_back_on_the_same_or_the_previous_day(toy):
    assert pick(toy, "C", "D", "2022/09/01 11:00")["reverse_payment_2d"] == 0  # paid back later
    assert pick(toy, "D", "C", "2022/09/02 11:00")["reverse_payment_2d"] == 1
    assert pick(toy, "E", "F", "2022/09/01 12:00")["reverse_payment_2d"] == 0
    assert pick(toy, "F", "E", "2022/09/05 12:00")["reverse_payment_2d"] == 0  # 4 days apart


def test_history_is_a_daily_average_over_the_days_the_file_covers(toy):
    """Two days of history on 3 September, seven on the 8th: one payment to one payee a day
    reads 1.0 on both, not 2 and 7."""
    for ts in ("2022/09/03 13:00", "2022/09/08 13:00"):
        row = pick(toy, "G", "H", ts)
        assert row["src_out_cnt_7d_avg"] == 1.0
        assert row["src_out_peers_7d_avg"] == 1.0
        assert row["dst_in_cnt_7d_avg"] == 1.0
        assert row["dst_in_peers_7d_avg"] == 1.0


def test_history_counts_whole_days_in_any_time_zone(toy):
    """In London, midnight on 24 October is 7 days and 1 hour before midnight on the 31st;
    the payment on the 24th must still count for the 31st."""
    assert pick(toy, "J", "K", "2022/10/31 09:00")["src_out_cnt_7d_avg"] == pytest.approx(1 / 7)


def test_pass_through_compares_us_dollar_in_and_out(toy):
    inflow = pick(toy, "L", "M", "2022/09/03 14:00")
    assert inflow["paid_usd"] == pytest.approx(670.0)
    assert inflow["received_usd"] == pytest.approx(670.0)
    assert inflow["log_amount"] == pytest.approx(np.log1p(670.0))
    assert inflow["dst_pass_through_day"] == pytest.approx(1.0)


def test_rates_use_only_rows_up_to_the_day(toy):
    """1.0 until Euro is paired with US Dollar, then the median of the rates seen so far."""
    paid = [pick(toy, "P", "Q", f"2022/09/0{d} 17:00")["paid_usd"] for d in (1, 2, 3)]
    assert paid == pytest.approx([100.0, 117.0, 118.5])
