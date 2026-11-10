"""The AMLworld transaction file layout (IBM, Kaggle release) and how this repo names it.

The Kaggle CSV header repeats the column name "Account" for sender and receiver, so files
are always read with these names in this order and the header line is skipped.
"""

from __future__ import annotations

RAW_HEADER = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]
COLUMNS = [
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
    "is_laundering",
]
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
PAYMENT_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
CURRENCIES = [
    "US Dollar", "Euro", "Yuan", "Yen", "UK Pound", "Rupee", "Ruble", "Bitcoin",
    "Canadian Dollar", "Australian Dollar", "Swiss Franc", "Mexican Peso", "Brazil Real",
    "Shekel", "Saudi Riyal",
]  # fmt: skip
PATTERN_TYPES = [
    "FAN-OUT", "FAN-IN", "CYCLE", "GATHER-SCATTER", "SCATTER-GATHER", "STACK", "RANDOM",
    "BIPARTITE",
]  # fmt: skip
