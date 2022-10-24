import numpy as np
import pandas as pd
import pytest

from spark_forecast.helpers._pyspark.time_series import SparkTimeSeriesHelper


TEST_DATA = [
    (  # 2022-01-01 is Saturday
        pd.Series(
            np.random.randint(100, size=100),
            index=pd.date_range("2022-01-01", periods=100, freq="D"),
        ),
        # The first monday should be 2021-12-27
        pd.date_range("2021-12-27", periods=15, freq="7D"),
    ),
    (  # 2022-08-02 is Tuesday
        pd.Series(
            np.random.randint(100, size=50),
            index=pd.date_range("2022-08-02", periods=50, freq="D"),
        ),
        # The first monday should be 2022-08-01
        pd.date_range("2022-08-01", periods=8, freq="7D"),
    ),
]


@pytest.mark.parametrize("serie, expected_index", TEST_DATA)
def test_build_weekly_aggregation(serie, expected_index):

    serie_weekly = SparkTimeSeriesHelper.build_weekly_aggregation(
        serie, day="MON"
    )
    assert serie.sum() == serie_weekly.sum()
    assert serie_weekly.index.equals(expected_index)
