# -*- coding: utf-8 -*-
"""Tests for outliers outlier removal"""

import numpy as np
import pandas as pd
import pytest


from spark_forecast.helpers._pyspark.time_series import SparkTimeSeriesHelper
from spark_forecast.helpers._pyspark.time_series import OUTLIERS_METHODS


TEST_DATA = [
    pd.DataFrame(
        {
            "A": pd.concat(
                [
                    pd.Series(np.random.randint(100, size=100)),
                    pd.Series([10000, 20000]),
                ],
                ignore_index=True,
            ),
            "B": np.ones(102),
            "item_id": 1,
            "ds": pd.date_range("2022-01-01", periods=102, freq="D"),
        },
    ),
    pd.DataFrame(
        {
            "A": pd.concat(
                [
                    pd.Series(np.random.randint(100, size=100)),
                    pd.Series([10000, 20000]),
                ],
                ignore_index=True,
            ),
            "B": np.ones(102),
            "item_id": 1,
            "ds": pd.date_range("2022-01-01", periods=102, freq="D"),
        },
    ),
]


@pytest.mark.parametrize("data", TEST_DATA)
def test_outliers_removal_defaults(data):

    helper = SparkTimeSeriesHelper(
        freq="D",
        target_column="A",
        time_column="ds",
        group_columns="item_id",
    )

    for method in OUTLIERS_METHODS:
        fixed_data = helper.fill_and_fix_time_series_from_grouped_data(
            data,
            duplicate_strategy="mode",
            fillna_strategy="zero",
            interpolator={"method": "linear"},
            outliers_detection={"method": method},
        )

        assert fixed_data.A.max() <= 100
        assert fixed_data.A.min() >= 0
