# -*- coding: utf-8 -*-
"""Test for ``spark_forecast.preprocessing.utils``"""


import pandas as pd
import pytest

from spark_forecast.preprocessing.utils import DatesFeatures

# nosec

DAY_TIME_INDEXES = [
    pd.date_range("2018-01-01", periods=367),
    pd.date_range("2018-06-01", periods=367),
    pd.date_range("2002-12-25", periods=500),
    pd.date_range("2022-06-25", periods=500),
]


@pytest.mark.parametrize("date_index", DAY_TIME_INDEXES)
def test_day_freq(date_index):
    dates_features_maker = DatesFeatures(freq="D")
    dates_features = dates_features_maker.create_dates_features(
        date_index, extract_columns=["month", "week", "dayofweek"]
    )

    assert sorted(list(dates_features.month.unique())) == list(range(1, 13))
    assert sorted(list(dates_features.week.unique())) == list(range(0, 3))
    assert sorted(list(dates_features.dayofweek.unique())) == list(range(0, 7))


######## ADDED BY FM 2022-07-13
@pytest.mark.parametrize("date_index", DAY_TIME_INDEXES)
def test_polar_date_features(date_index):
    """test to check the function create_dates_features_polar"""

    dates_features_maker = DatesFeatures(freq="D")
    dates_features = dates_features_maker.create_dates_features_polar(
        date_index,
        extract_columns=[
            "month",
            "week",
            "weekofyear",
            "dayofweek",
            "dayofmonth",
        ],
    )

    assert list(dates_features.columns) == [
        "month_x",
        "month_y",
        "week_x",
        "week_y",
        "weekofyear_x",
        "weekofyear_y",
        "dayofweek_x",
        "dayofweek_y",
        "dayofmonth_x",
        "dayofmonth_y",
    ]
    assert (dates_features.max().max() <= 1) & (
        dates_features.min().min() >= -1
    )
