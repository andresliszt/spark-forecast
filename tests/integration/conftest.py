# -*- coding: utf-8 -*-
"""Conftest models."""

from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from spark_forecast import SETTINGS


@pytest.fixture(
    scope="module",
    params=list(
        Path(
            SETTINGS.PROJECT_PATH,
            "tests",
            "integration",
            "data",
            "time_series",
        ).iterdir()
    ),
)
def test_file(request):
    yield request.param


@pytest.fixture(scope="module")
def setup_spark():
    """Fixture to set up the in-memory spark with test data"""

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("fairy-forecast-test")
        .getOrCreate()
    )
    yield spark


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="module")
def pyspark_test_data(setup_spark, test_file):
    """Fixture to set up the in-memory pyspark pandas Data frame test data"""
    yield setup_spark.read.csv(
        str(test_file),
        header=True,
        schema="location_id string, item_id string, ds date, y double, VentaNeta double, inv_init double, precio double",
    )
