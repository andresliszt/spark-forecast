# -*- coding: utf-8 -*-
"""Conftest models."""

from pathlib import Path

import pandas as pd
import pytest

from spark_forecast import SETTINGS


@pytest.fixture(
    scope="module",
    params=list(
        Path(
            SETTINGS.PROJECT_PATH, "tests", "unit", "data", "test_data_clean"
        ).iterdir()
    ),
)
def test_file_pandas(request):
    yield pd.read_csv(request.param)


@pytest.fixture(
    scope="module",
    params=list(
        Path(
            SETTINGS.PROJECT_PATH,
            "tests",
            "unit",
            "data",
            "test_data_forecast",
        ).iterdir()
    ),
)
def test_file_forecasting_pandas(request):
    """Files to test forecast methods

    Those files must **perfect** i.e
    without nulls, missing dates, etc.

    And each csv is a different single
    time series.

    """
    yield pd.read_csv(request.param)
