# -*- coding: utf-8 -*-
"""Tests for fillers and interpolators on time series."""

import numpy as np
import pandas as pd
import pytest


from spark_forecast.helpers._pyspark.time_series import SparkTimeSeriesHelper
from spark_forecast.preprocessing.utils import (
    check_is_datetime_indexed_and_frequency,
)

TEST_DATA = [
    (
        pd.Series(
            [1, 1, 1, 1, 1, 1],
            index=pd.to_datetime(
                [
                    "2022-02-10",
                    "2022-02-11",
                    "2022-02-14",
                    "2022-02-14",
                    "2022-02-24",
                    "2022-02-16",
                ]
            ),
        ),
        pd.date_range("2022-02-10", "2022-02-24"),
        {
            "freq": "D",
            "duplicate_strategy": "mode",
            "interpolator": None,
            "fillna_strategy": "zero",
        },
    ),
    (
        pd.Series(
            [1, 1, 1, 1, 1, 1],
            index=pd.to_datetime(
                [
                    "2022-02-10",
                    "2022-02-11",
                    "2022-02-14",
                    "2022-02-14",
                    "2022-02-24",
                    "2022-02-16",
                ]
            ),
        ),
        pd.date_range("2022-02-10", "2022-02-24"),
        {
            "freq": "D",
            "duplicate_strategy": "mean",
            "interpolator": {"method": "linear"},
            "fillna_strategy": "zero",
        },
    ),
    (
        pd.Series(
            [np.nan, np.nan, 1, 1, 1, 1],
            index=pd.to_datetime(
                [
                    "2022-01-01 04:00:00",
                    "2022-01-01 05:00:00",
                    "2022-01-01 00:00:00",
                    "2022-01-01 06:00:00",
                    "2022-01-01 03:00:00",
                    "2022-01-01 00:00:00",
                ]
            ),
        ),
        pd.date_range("2022-01-01 00:00:00", periods=7, freq="H"),
        {
            "freq": "H",
            "duplicate_strategy": "median",
            "interpolator": {"method": "linear"},
            "fillna_strategy": "median",
        },
    ),
]


@pytest.mark.parametrize("serie, date_range, method_kwargs", TEST_DATA)
def test_complete_dates_in_time_series(serie, date_range, method_kwargs):

    serie = SparkTimeSeriesHelper.fill_and_fix_time_series(
        serie, **method_kwargs
    )
    check_is_datetime_indexed_and_frequency(serie, freq=method_kwargs["freq"])
    assert (serie.index == date_range).all()
    assert serie.isnull().sum() == 0


KWARGS_FOR_TEST_FILES = [
    {
        "duplicate_strategy": "median",
        "interpolator": {"method": "linear"},
        "fillna_strategy": "median",
        "outliers_detection": None,
    },
    {
        "duplicate_strategy": "median",
        "interpolator": {"method": "linear"},
        "fillna_strategy": "median",
        "outliers_detection": {"method": "iqr"},
    },
    {
        "duplicate_strategy": "median",
        "interpolator": {"method": "linear"},
        "fillna_strategy": "mean",
        "outliers_detection": None,
    },
    {
        "duplicate_strategy": "median",
        "interpolator": {"method": "linear"},
        "fillna_strategy": "mean",
        "outliers_detection": {"method": "iqr"},
    },
    {
        "duplicate_strategy": "median",
        "interpolator": {"method": "linear"},
        "fillna_strategy": "zero",
        "outliers_detection": None,
    },
    {
        "duplicate_strategy": "median",
        "interpolator": {"method": "linear"},
        "fillna_strategy": "zero",
        "outliers_detection": {"method": "iqr"},
    },
]


@pytest.mark.parametrize("kwargs", KWARGS_FOR_TEST_FILES)
def test_complete_dates_in_time_series_from_csv(test_file_pandas, kwargs):

    time_helper = SparkTimeSeriesHelper(
        freq="D",
        target_column="y",
        time_column="ds",
        group_columns=["location_id", "item_id"],
    )

    data = time_helper.fill_and_fix_time_series_from_grouped_data(
        test_file_pandas, **kwargs
    )
    y = pd.Series(data["y"].values, index=pd.to_datetime(data["ds"]))
    check_is_datetime_indexed_and_frequency(y, freq="D")
    assert y.isnull().sum() == 0
