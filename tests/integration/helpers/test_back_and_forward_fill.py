# -*- coding: utf-8 -*-
"""pandas API integration tests."""

import datetime

import pandas as pd

from spark_forecast.helpers._pyspark.time_series import SparkTimeSeriesHelper
from spark_forecast.preprocessing.utils import (
    check_is_datetime_indexed_and_frequency,
)


def __check_is_back_and_forward_filled(
    data_grouped: pd.DataFrame, min_date: str, max_date: str
) -> None:
    """Check if the data comming from group by operation is a suitable series"""
    _min = data_grouped["ds"].min()
    _max = data_grouped["ds"].max()
    data_grouped["ds"] = pd.to_datetime(data_grouped["ds"])
    data_grouped = data_grouped.set_index("ds")
    check_is_datetime_indexed_and_frequency(data_grouped["y"], freq="D")

    assert not data_grouped["y"].isnull().any()
    assert min_date == _min
    assert max_date == _max


def test_complete_dates_in_time_series_in_pandas_pyspark(
    pyspark_test_data,
):

    helper = SparkTimeSeriesHelper(
        freq="D",
        target_column="y",
        time_column="ds",
        group_columns=["location_id", "item_id"],
    )

    # This test is valid only if each series
    # whose time index es strctly between '2005-01-01' and '2030-01-01'

    pyspark_test_data = helper.fill_and_fix_time_series_spark(
        pyspark_test_data,
        duplicate_strategy="mode",
        fillna_strategy="zero",
        interpolator={"method": "linear"},
        outliers_detection={"method": "iqr"},
        repeat_bfill_to="2005-01-01",
        repeat_ffill_to="2030-01-01",
    )
    pyspark_test_data = pyspark_test_data.toPandas()

    pyspark_test_data.groupby(["location_id", "item_id"]).apply(
        __check_is_back_and_forward_filled,
        min_date=datetime.date(2005, 1, 1),
        max_date=datetime.date(2030, 1, 1),
    )
