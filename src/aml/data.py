"""Get the AMLworld files and read their laundering patterns.

Two ways to get the data, both recording each file's SHA-256 in data/raw/manifest.json:
  - `aml download`: fetches from Kaggle with KAGGLE_USERNAME and KAGGLE_KEY taken from the
    environment (never from a file in this repo, never printed);
  - `aml import PATH`: copies a file you downloaded yourself from the Kaggle page, or
    unpacks the files of one dataset from a zip you downloaded there.
A file in data/raw is replaced only once its new copy is complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from aml.schema import PATTERN_TYPES

KAGGLE_DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
KAGGLE_PAGE = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
KAGGLE_DOWNLOAD = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}/{{name}}"
DATASETS = ("HI-Small", "LI-Small", "HI-Medium", "LI-Medium", "HI-Large", "LI-Large")
# Kaggle's main activity period of each size: Small Sep 1 to 10, Medium Sep 1 to 16,
# Large Aug 1 to Nov 5, 2022. Each value is the first day after that period and is an
# exclusive cutoff (features.build keeps rows with ts < PERIOD_END), so rows on or after
# that day are dropped. Those later transactions are all laundering (see the Kaggle
# discussion) and would leak the label through the timestamp.
PERIOD_END = {"Small": "2022-09-11", "Medium": "2022-09-17", "Large": "2022-11-06"}

NOTICE = f"""\
IBM AMLworld data notice
========================
Source:  {KAGGLE_PAGE}
License: Community Data License Agreement, Sharing, version 1.0 (CDLA-Sharing-1.0)
Content: fully synthetic transactions from a simulated world; no real people.
Files are kept in data/raw on your machine and are never committed.
"""


class DataFileError(ValueError):
    """A file that is not the AMLworld file it should be."""


def raw_dir(data_dir: Path) -> Path:
    return data_dir / "raw"


def files_for(dataset: str) -> list[str]:
    return [f"{dataset}_Trans.csv", f"{dataset}_Patterns.txt"]


# Every file name the pipeline reads; SYN is the generated dataset that tests and CI use.
EXPECTED = {name for d in (*DATASETS, "SYN") for name in files_for(d)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(data_dir: Path) -> Path:
    return raw_dir(data_dir) / "manifest.json"


def record(data_dir: Path, path: Path) -> str:
    """Record the file's SHA-256 in the manifest and return it."""
    manifest = _manifest(data_dir)
    entries = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    entries[path.name] = sha256(path)
    manifest.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return entries[path.name]


def import_file(data_dir: Path, source: Path, dataset: str = "HI-Small") -> list[tuple[Path, str]]:
    """Copy one AMLworld file into data/raw, or unpack the `dataset` files from a zip.
    Return each file written with its SHA-256."""
    if not source.is_file():
        raise FileNotFoundError(f"{source} not found")
    dest_dir = raw_dir(data_dir)
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:  # read from disk, never loaded whole
            held = {Path(m).name for m in archive.namelist()}
            names = [n for n in files_for(dataset) if n in held]
            if not names:
                raise DataFileError(
                    f"{source.name} holds no {dataset} files. Pass --dataset to pick the "
                    f"dataset to take from it (one of {', '.join(DATASETS)})."
                )
            dest_dir.mkdir(parents=True, exist_ok=True)
            paths = [_extract_member(archive, n, dest_dir) for n in names]
    else:
        if source.name not in EXPECTED:
            raise DataFileError(
                f"{source.name} is not an AMLworld file name. Expected <dataset>_Trans.csv "
                f"or <dataset>_Patterns.txt for one of {', '.join(DATASETS)}; remove any "
                "browser suffix such as ' (1)'."
            )
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        if source.resolve() != dest.resolve():
            tmp = dest_dir / f"{source.name}.tmp"
            try:
                shutil.copyfile(source, tmp)
                os.replace(tmp, dest)
            finally:
                tmp.unlink(missing_ok=True)
        paths = [dest]
    return [(p, record(data_dir, p)) for p in paths]


def _extract_member(archive: zipfile.ZipFile, name: str, dest_dir: Path) -> Path:
    """Copy the one member whose base name is `name` to dest_dir/name. Only that member is
    read, wherever it sits in the archive, and nothing in the archive decides where a file
    is written. The real file is replaced only once the new copy is complete."""
    members = [m for m in archive.namelist() if Path(m).name == name]
    if len(members) != 1:
        raise DataFileError(f"archive does not hold exactly one {name}")
    dest, tmp = dest_dir / name, dest_dir / f"{name}.tmp"
    try:
        with archive.open(members[0]) as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)  # reading to the end checks the CRC
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _extract_single(part: Path, name: str, dest_dir: Path) -> Path:
    """Kaggle may send the file itself or a zip holding it. `part` is the finished
    download; the real file is replaced only once the new copy is complete."""
    with part.open("rb") as handle:
        is_zip = handle.read(2) == b"PK"
    if not is_zip:
        dest = dest_dir / name
        os.replace(part, dest)
        return dest
    with zipfile.ZipFile(part) as archive:
        return _extract_member(archive, name, dest_dir)


def download(
    data_dir: Path, dataset: str = "HI-Small", *, transport: httpx.BaseTransport | None = None
) -> list[tuple[Path, str]]:
    """Fetch the dataset's files from Kaggle and return each with its SHA-256. Each file
    is streamed to disk, so a large one never sits whole in memory."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        raise PermissionError(
            "Set KAGGLE_USERNAME and KAGGLE_KEY in your environment (from your Kaggle account "
            f"settings), or download the files from {KAGGLE_PAGE} and run `aml import FILE`."
        )
    dest_dir = raw_dir(data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    # httpx drops the credentials when Kaggle redirects to its storage host.
    with httpx.Client(
        timeout=600.0, follow_redirects=True, auth=(user, key), transport=transport
    ) as http:
        for name in files_for(dataset):
            part = dest_dir / f"{name}.part"
            try:
                with http.stream("GET", KAGGLE_DOWNLOAD.format(name=name)) as response:
                    if response.status_code in (401, 403):
                        raise PermissionError(
                            f"Kaggle refused the request for {name} (HTTP "
                            f"{response.status_code}). Check KAGGLE_USERNAME and KAGGLE_KEY, "
                            f"or download the files from {KAGGLE_PAGE} and run "
                            "`aml import FILE`."
                        )
                    response.raise_for_status()
                    with part.open("wb") as handle:
                        for chunk in response.iter_bytes(1 << 20):
                            handle.write(chunk)
                path = _extract_single(part, name, dest_dir)
            finally:
                part.unlink(missing_ok=True)
            out.append((path, record(data_dir, path)))
    return out


@dataclass(frozen=True)
class PatternTxn:
    kind: str
    attempt: int
    key: tuple[str, str, str, str, str, str]  # timestamp, banks, accounts, amount paid


def read_patterns(path: Path) -> list[PatternTxn]:
    """Parse <dataset>_Patterns.txt: transactions grouped by laundering attempt and type.
    A header names the type bare ("BEGIN LAUNDERING ATTEMPT - CYCLE") or with details
    after a colon, as in IBM's files. A type outside schema.PATTERN_TYPES stops the run,
    because it means the file format has changed."""
    out: list[PatternTxn] = []
    kind, attempt = None, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            attempt += 1
            kind = line.split("-", 1)[1].split(":")[0].strip()
            if kind not in PATTERN_TYPES:
                raise ValueError(f"unknown laundering pattern {kind!r} in {path.name}: {line!r}")
        elif line.startswith("END LAUNDERING ATTEMPT"):
            kind = None
        elif line and kind:
            f = line.split(",")
            out.append(PatternTxn(kind, attempt, (f[0], f[1], f[2], f[3], f[4], f[7])))
    return out


def size_of(dataset: str) -> str:
    return dataset.split("-", 1)[1]
