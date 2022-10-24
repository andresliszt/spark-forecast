import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spark_forecast.exc import CovariatesWithNullValues
from spark_forecast.exc import InvalidDateTimeFrequency
from spark_forecast.exc import NotDateTimeIndexError
from spark_forecast.exc import TimeSeriesAndCovariatesIndexError
from spark_forecast.exc import TooShortTimeSeries
from spark_forecast.models.regressor import RegressorForecast

# Each element of TEST_DATA_LAGS is a 5-tuple where
# t[0] is the number of lags, t[1] is the time series
# with a defined frequency and time index
# t[2] is the exogenous variables only for train (pd.DF or pd.Series)
# t[3] is y_train
# t[4] the expected X_train (length len(series) - number of lags)

TEST_DATA_LAGS = [
    (
        3,
        pd.Series(
            [1, 2, 3, 4, 5],
            index=pd.date_range("2022-01-01", periods=5, freq="D"),
        ),
        pd.DataFrame(
            {"A": [1, 2, 3, 4, 5], "B": [True, True, False, False, False]},
            index=pd.date_range("2022-01-01", periods=5, freq="D"),
        ),
        pd.Series([4.0, 5.0], index=pd.date_range("2022-01-04", "2022-01-05")),
        pd.DataFrame(
            {
                "lag_1": [3.0, 4.0],
                "lag_2": [2.0, 3.0],
                "lag_3": [1.0, 2.0],
                "A": [4.0, 5.0],
                "B": [0.0, 0.0],
            },
            index=pd.date_range("2022-01-04", "2022-01-05"),
        ),
    ),
    (
        3,
        pd.Series(
            [1, 1, 1, 1, 2, 3, 4, 5],
            index=pd.date_range("2022-01-01", periods=8, freq="D"),
        ),
        pd.DataFrame(
            {
                "A": [1, 1, 0, 0, 0, 1, 2, 0],
                "B": [True, False, False, True, True, False, False, False],
            },
            index=pd.date_range("2022-01-01", periods=8, freq="D"),
        ),
        pd.Series(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            index=pd.date_range("2022-01-04", periods=5, freq="D"),
        ),
        pd.DataFrame(
            {
                "lag_1": [1.0, 1.0, 2.0, 3.0, 4.0],
                "lag_2": [1.0, 1.0, 1.0, 2.0, 3.0],
                "lag_3": [1.0, 1.0, 1.0, 1.0, 2.0],
                "A": [0.0, 0.0, 1.0, 2.0, 0.0],
                "B": [1.0, 1.0, 0.0, 0.0, 0.0],
            },
            index=pd.date_range("2022-01-04", periods=5, freq="D"),
        ),
    ),
    (
        5,
        pd.Series(
            [1, 1, 1, 1, 2, 3, 4, 5],
            index=pd.date_range("2022-01-01", periods=8, freq="D"),
        ),
        pd.Series(
            [True, True, True, True, True, False, False, False],
            name="A",
            index=pd.date_range("2022-01-01", periods=8, freq="D"),
        ),
        pd.Series(
            [3.0, 4.0, 5.0],
            index=pd.date_range("2022-01-06", periods=3, freq="D"),
        ),
        pd.DataFrame(
            {
                "lag_1": [2.0, 3.0, 4.0],
                "lag_2": [1.0, 2.0, 3.0],
                "lag_3": [1.0, 1.0, 2.0],
                "lag_4": [1.0, 1.0, 1.0],
                "lag_5": [1.0, 1.0, 1.0],
                "A": [0.0, 0.0, 0.0],
            },
            index=pd.date_range("2022-01-06", periods=3, freq="D"),
        ),
    ),
    (
        7,
        pd.Series(
            [2, 3, 4, 5, 10, 30, 20, 12],
            index=pd.date_range("2022-01-01", periods=8, freq="D"),
        ),
        pd.DataFrame(
            {
                "A": [1, 1, 0, 0, 0, 1, 2, 0],
                "B": [True, False, False, True, True, False, False, False],
                "C": [1, 0, 3, 0, 0, 1, 1, 4],
            },
            index=pd.date_range("2022-01-01", periods=8, freq="D"),
        ),
        pd.Series(
            [12.0], index=pd.date_range("2022-01-08", periods=1, freq="D")
        ),
        pd.DataFrame(
            {
                "lag_1": 20.0,
                "lag_2": 30.0,
                "lag_3": 10.0,
                "lag_4": 5.0,
                "lag_5": 4.0,
                "lag_6": 3.0,
                "lag_7": 2.0,
                "A": 0.0,
                "B": 0.0,
                "C": 4.0,
            },
            index=pd.date_range("2022-01-08", periods=1, freq="D"),
        ),
    ),
    (
        8,
        pd.Series(
            [0, 2, 1, 1, 2, 3, 4, 5, 10, 30, 20, 12],
            index=pd.date_range("2022-01-01", periods=12, freq="D"),
        ),
        pd.Series(
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                False,
            ],
            name=None,
            index=pd.date_range("2022-01-01", periods=12, freq="D"),
        ),
        pd.Series(
            [10.0, 30.0, 20.0, 12.0],
            index=pd.date_range("2022-01-09", periods=4, freq="D"),
        ),
        pd.DataFrame(
            {
                "lag_1": [5.0, 10.0, 30.0, 20.0],
                "lag_2": [4.0, 5.0, 10.0, 30.0],
                "lag_3": [3.0, 4.0, 5.0, 10.0],
                "lag_4": [2.0, 3.0, 4.0, 5.0],
                "lag_5": [1.0, 2.0, 3.0, 4.0],
                "lag_6": [1.0, 1.0, 2.0, 3.0],
                "lag_7": [2.0, 1.0, 1.0, 2.0],
                "lag_8": [0.0, 2.0, 1.0, 1.0],
                0: [1.0, 1.0, 1.0, 0.0],
            },
            index=pd.date_range("2022-01-09", periods=4, freq="D"),
        ),
    ),
]


@pytest.mark.parametrize(
    "lags, serie, future_covariates, expected_y_train, expected_X_train",
    TEST_DATA_LAGS,
)
def test_data_lags(
    lags, serie, future_covariates, expected_y_train, expected_X_train
):
    forecaster = RegressorForecast(
        regressor=LinearRegression(),
        lags=lags,
        freq=serie.index.freq,
    )
    X_train, y_train = forecaster.create_train_set(
        serie, future_covariates=future_covariates
    )

    assert X_train.eq(expected_X_train).all().all()
    assert X_train.index.equals(expected_X_train.index)
    assert y_train.eq(expected_y_train).all()
    assert y_train.index.equals(expected_y_train.index)


TEST_WRONG_DATA = [
    (  # This error is Because lags >= len(serie)
        (5, "D"),
        pd.Series(
            [1, 2, 3, 4, 5],
            index=pd.date_range("2022-01-01", periods=5, freq="D"),
        ),
        None,
        TooShortTimeSeries,
    ),
    (  # This error is Because lags is float
        (3.3, "D"),
        pd.Series(
            [1, 2, 3, 4, 5],
            index=pd.date_range("2022-01-01", periods=5, freq="D"),
        ),
        None,
        ValueError,
    ),
    (  # This error is Because lags < 1
        (-1, "D"),
        pd.Series(
            [1, 2, 3, 4, 5],
            index=pd.date_range("2022-01-01", periods=5, freq="D"),
        ),
        None,
        ValueError,
    ),
    (  # This error is Because serie hasn't datetime index
        (2, "D"),
        pd.Series(
            [1, 2, 3, 4, 5],
        ),
        None,
        NotDateTimeIndexError,
    ),
    (  # This error is Because serie hasn't valid frequency
        (2, "D"),
        pd.Series(
            [1, 2, 3, 4, 5],
            index=pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-01-02",
                    "2022-01-04",
                    "2022-01-03",
                    "2022-01-11",
                ]
            ),
        ),
        None,
        InvalidDateTimeFrequency,
    ),
    (  # This error is Because serie's index and covariates index differ
        (2, "D"),
        pd.Series(
            [1, 2, 3, 4, 5],
            index=pd.date_range("2022-01-01", periods=5, freq="D"),
        ),
        pd.Series(
            [True, True, True, True, True],
            name="A",
            index=pd.date_range("2022-02-01", periods=5, freq="D"),
        ),
        TimeSeriesAndCovariatesIndexError,
    ),
    (  # This error is Because covariates has null values
        (2, "D"),
        pd.Series(
            [1, 2, 3, 4, 5],
            index=pd.date_range("2022-01-01", periods=5, freq="D"),
        ),
        pd.Series(
            [True, True, True, np.nan, True],
            name="A",
            index=pd.date_range("2022-01-01", periods=5, freq="D"),
        ),
        CovariatesWithNullValues,
    ),
]


@pytest.mark.parametrize(
    "lags_and_freq, serie, future_covariates, error", TEST_WRONG_DATA
)
def test_wrong_execution(lags_and_freq, serie, future_covariates, error):

    with pytest.raises(error):
        forecaster = RegressorForecast(
            regressor=LinearRegression(),
            lags=lags_and_freq[0],
            freq=lags_and_freq[1],
        )
        forecaster.create_train_set(serie, future_covariates=future_covariates)
