# -*- coding: utf-8 -*-
"""pandas API integration tests."""

import pandas as pd

from spark_forecast.helpers._pyspark.time_series import SparkTimeSeriesHelper
from spark_forecast.preprocessing.utils import (
    check_is_datetime_indexed_and_frequency,
)


def __check_is_filled(data_grouped: pd.DataFrame) -> None:
    """Check if the data comming from group by operation is a suitable series"""
    data_grouped["ds"] = pd.to_datetime(data_grouped["ds"])
    data_grouped = data_grouped.set_index("ds")
    check_is_datetime_indexed_and_frequency(data_grouped["y"], freq="D")
    assert not data_grouped["y"].isnull().any()


def test_complete_dates_in_time_series_in_pandas_pyspark(
    pyspark_test_data,
):

    helper = SparkTimeSeriesHelper(
        freq="D",
        target_column="y",
        time_column="ds",
        group_columns=["location_id", "item_id"],
    )

    pyspark_test_data = helper.fill_and_fix_time_series_spark(
        pyspark_test_data,
        duplicate_strategy="mode",
        fillna_strategy="zero",
        interpolator={"method": "linear"},
        outliers_detection={"method": "iqr"},
    )
    pyspark_test_data = pyspark_test_data.toPandas()

    pyspark_test_data.groupby(["location_id", "item_id"]).apply(
        __check_is_filled
    )
