"""A small generator that writes files in the AMLworld layout, with planted laundering.

It exists so tests, CI and the Quickstart demo run without downloading the real data.
Background payments are random; some change currency at the fixed rates in USD_RATES, and
Reinvestment payments go from an account to itself (a self transfer). Laundering attempts
follow four of the AMLworld patterns (fan out, fan in, cycle, scatter gather), are
labelled, and are listed in a patterns file in the same format IBM uses, with headers both
bare and with details after a colon. It is a test fixture, not a substitute for AMLworld:
never quote numbers from it.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from aml.schema import CURRENCIES, PAYMENT_FORMATS, RAW_HEADER, TIMESTAMP_FORMAT

# US Dollar per unit of each currency the generator uses (CURRENCIES[:6]), fixed for the
# whole file, so tests know what a correct conversion gives.
USD_RATES = {
    "US Dollar": 1.0,
    "Euro": 1.17,
    "Yuan": 0.14,
    "Yen": 0.0067,
    "UK Pound": 1.33,
    "Rupee": 0.012,
}
CROSS_SHARE = 0.15  # background payments received in another currency
# Header details in IBM's own wording, kept as IBM writes them. SCATTER-GATHER stays bare,
# so the file has both header forms.
DETAILS = {
    "FAN-OUT": "Max 5-degree Fan-Out",
    "FAN-IN": "Max 5-degree Fan-In",
    "CYCLE": "Max 4 hops",
}


@dataclass(frozen=True)
class Txn:
    ts: datetime
    from_bank: int
    from_account: str
    to_bank: int
    to_account: str
    amount: float  # paid, in `currency`
    currency: str  # payment currency
    fmt: str
    laundering: int
    to_currency: str | None = None  # receiving currency, when it differs from `currency`

    def row(self) -> list[str]:
        to_currency = self.to_currency or self.currency
        received = (
            self.amount
            if to_currency == self.currency
            else self.amount * USD_RATES[self.currency] / USD_RATES[to_currency]
        )
        return [
            self.ts.strftime(TIMESTAMP_FORMAT),
            f"{self.from_bank:03d}",
            self.from_account,
            f"{self.to_bank:03d}",
            self.to_account,
            f"{received:.2f}",
            to_currency,
            f"{self.amount:.2f}",
            self.currency,
            self.fmt,
            str(self.laundering),
        ]


def generate(
    out_dir: Path,
    *,
    prefix: str = "SYN",
    accounts: int = 400,
    banks: int = 8,
    days: int = 10,
    background: int = 6000,
    attempts: int = 40,
    seed: int = 7,
) -> tuple[Path, Path]:
    """Write <prefix>_Trans.csv and <prefix>_Patterns.txt into out_dir."""
    rng = random.Random(seed)
    start = datetime(2022, 9, 1)
    ids = [(rng.randrange(1, banks + 1), f"{rng.getrandbits(36):09X}") for _ in range(accounts)]
    currencies = CURRENCIES[:6]  # the keys of USD_RATES

    def when(day: int | None = None) -> datetime:
        d = rng.randrange(days) if day is None else day
        return start + timedelta(days=d, minutes=rng.randrange(24 * 60))

    txns: list[Txn] = []
    for _ in range(background):
        (fb, fa), (tb, ta) = rng.sample(ids, 2)
        amount = round(rng.lognormvariate(6.5, 1.6), 2)
        currency, fmt = rng.choice(currencies), rng.choice(PAYMENT_FORMATS)
        to_currency = None
        if rng.random() < CROSS_SHARE:
            to_currency = rng.choice([c for c in currencies if c != currency])
        if fmt == "Reinvestment":  # a self transfer: an account paying itself
            tb, ta, to_currency = fb, fa, None
        txns.append(Txn(when(), fb, fa, tb, ta, amount, currency, fmt, 0, to_currency))

    patterns: list[tuple[str, list[Txn]]] = []
    kinds = ["FAN-OUT", "FAN-IN", "CYCLE", "SCATTER-GATHER"]
    for i in range(attempts):
        kind = kinds[i % len(kinds)]
        day = rng.randrange(days)
        t0 = start + timedelta(days=day, minutes=rng.randrange(20 * 60))
        amount = round(rng.uniform(2000, 9500), 2)
        fmt = rng.choice(["ACH", "Wire", "Cash", "Bitcoin"])
        cur = "US Dollar"
        members = rng.sample(ids, 7)
        hub, others = members[0], members[1:]

        def t(k: int, start_at: datetime = t0) -> datetime:
            return start_at + timedelta(minutes=17 * k + rng.randrange(10))

        if kind == "FAN-OUT":
            steps = [Txn(t(k), *hub, *o, amount / 4, cur, fmt, 1) for k, o in enumerate(others[:5])]
        elif kind == "FAN-IN":
            steps = [Txn(t(k), *o, *hub, amount / 4, cur, fmt, 1) for k, o in enumerate(others[:5])]
        elif kind == "CYCLE":
            ring = [hub, *others[:3]]
            steps = [
                Txn(t(k), *ring[k], *ring[(k + 1) % len(ring)], amount * (0.98**k), cur, fmt, 1)
                for k in range(len(ring))
            ]
        else:  # SCATTER-GATHER: hub -> middles -> sink
            middles, sink = others[:4], others[4]
            steps = [Txn(t(k), *hub, *m, amount / 4, cur, fmt, 1) for k, m in enumerate(middles)]
            steps += [
                Txn(t(k + 5), *m, *sink, amount / 4 * 0.97, cur, fmt, 1)
                for k, m in enumerate(middles)
            ]
        txns.extend(steps)
        patterns.append((kind, steps))

    txns.sort(key=lambda x: x.ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    trans_path = out_dir / f"{prefix}_Trans.csv"
    with trans_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(RAW_HEADER)
        for x in txns:
            writer.writerow(x.row())
    patterns_path = out_dir / f"{prefix}_Patterns.txt"
    with patterns_path.open("w", encoding="utf-8", newline="\n") as handle:
        for kind, steps in patterns:
            head = f"{kind}:  {DETAILS[kind]}" if kind in DETAILS else kind
            handle.write(f"BEGIN LAUNDERING ATTEMPT - {head}\n")
            for x in steps:
                handle.write(",".join(x.row()) + "\n")
            handle.write(f"END LAUNDERING ATTEMPT - {kind}\n\n")
    return trans_path, patterns_path
