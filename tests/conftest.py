"""Shared fixtures: a generated dataset in the AMLworld layout and one Spark session."""

from __future__ import annotations

from pathlib import Path

import pytest

from aml.data import PERIOD_END
from aml.synth import generate


@pytest.fixture(scope="session")
def syn_files(tmp_path_factory) -> tuple[Path, Path]:
    return generate(tmp_path_factory.mktemp("raw"), prefix="SYN", seed=11)


@pytest.fixture(scope="session")
def spark():
    from aml.features import spark_session

    session = spark_session("aml-tests", cores=2)
    session.conf.set("spark.sql.shuffle.partitions", "2")  # the test files are tiny
    yield session
    session.stop()


@pytest.fixture(scope="session")
def frame(spark, syn_files):
    from aml.features import build_frame

    # synth writes Sep 1 to 10, the same window as the Small datasets
    return build_frame(syn_files[0], period_end=PERIOD_END["Small"], spark=spark)
