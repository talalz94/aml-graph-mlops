import base64
import csv
import functools
import io
import zipfile

import httpx
import pytest

from aml import data
from aml.cli import main
from aml.schema import RAW_HEADER


def test_generated_file_matches_the_amlworld_layout(syn_files):
    trans, patterns = syn_files
    with trans.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        assert next(reader) == RAW_HEADER  # "Account" appears twice, like the real file
        rows = list(reader)
    assert sum(r[-1] == "1" for r in rows) > 0
    assert any(r[1:3] == r[3:5] for r in rows)  # self transfers, as in AMLworld
    assert any(r[6] != r[8] for r in rows)  # payments that change currency
    parsed = data.read_patterns(patterns)
    assert {p.kind for p in parsed} == {"FAN-OUT", "FAN-IN", "CYCLE", "SCATTER-GATHER"}
    assert len({p.attempt for p in parsed}) == 40
    heads = [x for x in patterns.read_text(encoding="utf-8").splitlines() if x.startswith("BEGIN")]
    assert any(":" in h for h in heads) and any(":" not in h for h in heads)  # both forms


def test_import_records_the_checksum(tmp_path, syn_files):
    [(path, digest)] = data.import_file(tmp_path, syn_files[1])
    manifest = (tmp_path / "raw" / "manifest.json").read_text(encoding="utf-8")
    assert digest == data.sha256(path)
    assert digest in manifest


@pytest.mark.parametrize("name", ["HI-Small_Trans (1).csv", "manifest.json", "notes.txt"])
def test_import_rejects_files_that_are_not_amlworld_files(tmp_path, name):
    source = tmp_path / name
    source.write_text("x\n", encoding="utf-8")
    with pytest.raises(data.DataFileError):
        data.import_file(tmp_path / "data", source)
    assert not (tmp_path / "data" / "raw" / "manifest.json").exists()
    assert main(["--data-dir", str(tmp_path / "data"), "import", str(source)]) == 1


def test_import_unpacks_one_dataset_from_a_zip(tmp_path):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("HI-Small_Trans.csv", "Timestamp\n")
        z.writestr("HI-Small_Patterns.txt", "BEGIN LAUNDERING ATTEMPT - CYCLE\n")
        z.writestr("LI-Small_Trans.csv", "Timestamp\n")
    raw = tmp_path / "data" / "raw"
    out = data.import_file(tmp_path / "data", archive)
    assert [p.name for p, _ in out] == ["HI-Small_Trans.csv", "HI-Small_Patterns.txt"]
    assert sorted(p.name for p in raw.iterdir()) == [
        "HI-Small_Patterns.txt",
        "HI-Small_Trans.csv",
        "manifest.json",
    ]
    assert main(["--data-dir", str(tmp_path / "data"), "import", str(archive), "--dataset",
                 "LI-Small"]) == 0  # fmt: skip
    assert (raw / "LI-Small_Trans.csv").exists()
    with pytest.raises(data.DataFileError, match="--dataset"):
        data.import_file(tmp_path / "data", archive, "HI-Large")


def test_download_needs_credentials_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    with pytest.raises(PermissionError, match="aml import"):
        data.download(tmp_path)


def test_download_accepts_plain_and_zipped_files(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as z:
        z.writestr("HI-Small_Patterns.txt", "BEGIN LAUNDERING ATTEMPT - CYCLE\n")

    def handler(request):
        if request.url.path.endswith("_Trans.csv"):
            return httpx.Response(200, content=b"Timestamp\n")
        return httpx.Response(200, content=zipped.getvalue())

    out = data.download(tmp_path, transport=httpx.MockTransport(handler))
    paths = [p for p, _ in out]
    assert [p.name for p in paths] == ["HI-Small_Trans.csv", "HI-Small_Patterns.txt"]
    assert paths[1].read_text(encoding="utf-8").startswith("BEGIN")
    assert [digest for _, digest in out] == [data.sha256(p) for p in paths]
    # nothing half written is left behind
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == [
        "HI-Small_Patterns.txt",
        "HI-Small_Trans.csv",
        "manifest.json",
    ]


def test_refused_credentials_get_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    refused = httpx.MockTransport(lambda request: httpx.Response(401))
    with pytest.raises(PermissionError, match="KAGGLE_KEY"):
        data.download(tmp_path, transport=refused)


def test_a_failed_download_ends_with_a_short_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    missing = httpx.MockTransport(lambda request: httpx.Response(404))
    monkeypatch.setattr(data, "download", functools.partial(data.download, transport=missing))
    assert main(["--data-dir", str(tmp_path), "download"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: download failed:") and "404" in err


def test_download_never_offers_generated_data():
    with pytest.raises(SystemExit):
        main(["download", "--dataset", "SYN"])


def test_a_missing_file_names_the_command_that_makes_it(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path), "features", "--dataset", "SYN"]) == 1
    assert "Run `aml synth` first." in capsys.readouterr().err
    assert main(["--data-dir", str(tmp_path), "features"]) == 1
    assert "Run `aml download` or `aml import` first." in capsys.readouterr().err


def test_key_goes_only_to_kaggle_and_is_never_printed(tmp_path, monkeypatch, capsys):
    secret = "fake-kaggle-key-93f1c2"
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", secret)
    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers.get("Authorization")))
        if request.url.host == "www.kaggle.com":
            name = request.url.path.rsplit("/", 1)[-1]
            location = f"https://storage.googleapis.com/bucket/{name}?sig=x"
            return httpx.Response(302, headers={"Location": location})
        return httpx.Response(200, content=b"Timestamp\n")

    mock = httpx.MockTransport(handler)
    monkeypatch.setattr(data, "download", functools.partial(data.download, transport=mock))
    assert main(["--data-dir", str(tmp_path), "download"]) == 0
    basic = "Basic " + base64.b64encode(f"u:{secret}".encode()).decode()
    assert seen == [("www.kaggle.com", basic), ("storage.googleapis.com", None)] * 2
    out, err = capsys.readouterr()
    manifest = (tmp_path / "raw" / "manifest.json").read_text(encoding="utf-8")
    assert all(secret not in text for text in (out, err, manifest))


def test_member_path_in_archive_is_ignored(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    part = raw / "HI-Small_Trans.csv.part"
    with zipfile.ZipFile(part, "w") as z:
        z.writestr("../HI-Small_Trans.csv", "Timestamp\n")
    path = data._extract_single(part, "HI-Small_Trans.csv", raw)
    assert path == raw / "HI-Small_Trans.csv"
    assert path.read_text(encoding="utf-8") == "Timestamp\n"
    assert [p.name for p in tmp_path.iterdir()] == ["raw"]


def test_archive_without_the_expected_file_is_rejected(tmp_path):
    part = tmp_path / "x.part"
    with zipfile.ZipFile(part, "w") as z:
        z.writestr("../other.txt", "x")
    with pytest.raises(ValueError):
        data._extract_single(part, "HI-Small_Trans.csv", tmp_path)


def test_a_bad_archive_keeps_the_previous_file(tmp_path):
    good = tmp_path / "HI-Small_Trans.csv"
    good.write_text("old", encoding="utf-8")
    part = tmp_path / "x.part"
    with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as z:
        z.writestr("HI-Small_Trans.csv", "Timestamp\n" * 100)
    raw = bytearray(part.read_bytes())
    raw[raw.index(b"Timestamp")] ^= 0xFF  # break the CRC
    part.write_bytes(bytes(raw))
    with pytest.raises(zipfile.BadZipFile):
        data._extract_single(part, "HI-Small_Trans.csv", tmp_path)
    assert good.read_text(encoding="utf-8") == "old"
    assert not (tmp_path / "HI-Small_Trans.csv.tmp").exists()


def test_patterns_parser_reads_both_header_forms(tmp_path):
    row = "2022/09/01 05:14,001,AAA,002,BBB,10.00,US Dollar,9.50,US Dollar,ACH,1"
    lines = [
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out",
        row,
        "END LAUNDERING ATTEMPT - FAN-OUT",
        "",
        "BEGIN LAUNDERING ATTEMPT - STACK",
        row,
        "END LAUNDERING ATTEMPT - STACK",
        "",
    ]
    path = tmp_path / "X_Patterns.txt"
    path.write_bytes("\r\n".join(lines).encode())
    parsed = data.read_patterns(path)
    assert [(p.kind, p.attempt) for p in parsed] == [("FAN-OUT", 1), ("STACK", 2)]
    # received and paid differ, so this also shows the amount paid is the one used
    assert parsed[0].key == ("2022/09/01 05:14", "001", "AAA", "002", "BBB", "9.50")


def test_unknown_pattern_type_stops_the_run(tmp_path):
    path = tmp_path / "X_Patterns.txt"
    path.write_text("BEGIN LAUNDERING ATTEMPT - FAN OUT\n", encoding="utf-8")
    with pytest.raises(ValueError, match="FAN OUT"):
        data.read_patterns(path)
