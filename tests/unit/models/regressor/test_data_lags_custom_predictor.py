import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


from spark_forecast.models.regressor import RegressorForecastWithPredictors


TEST_DATA_LAGS_CUSTOM_PREDICTORS = [
    (
        (3, 4),
        pd.Series(
            [1, 2, 3, 4, 5],
            index=pd.date_range("2022-01-01", periods=5, freq="D"),
        ),
        pd.DataFrame(
            {"A": [1, 2, 3, 4, 5], "B": [True, True, False, False, False]},
            index=pd.date_range("2022-01-01", periods=5, freq="D"),
        ),
        pd.Series([5.0], index=pd.date_range("2022-01-05", "2022-01-05")),
        pd.DataFrame(
            {
                "lag_1": [4.0],
                "lag_2": [3.0],
                "lag_3": [2.0],
                "custom_predictor_1": [2.5],
                "A": [5.0],
                "B": [0.0],
            },
            index=pd.date_range("2022-01-05", periods=1),
        ),
    ),
    (
        (3, 5),
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
            [3.0, 4.0, 5.0],
            index=pd.date_range("2022-01-06", periods=3, freq="D"),
        ),
        pd.DataFrame(
            {
                "lag_1": [2.0, 3.0, 4.0],
                "lag_2": [1.0, 2.0, 3.0],
                "lag_3": [1.0, 1.0, 2.0],
                "custom_predictor_1": [1.2, 1.6, 2.2],
                "A": [1.0, 2.0, 0.0],
                "B": [0.0, 0.0, 0.0],
            },
            index=pd.date_range("2022-01-06", periods=3, freq="D"),
        ),
    ),
]


@pytest.mark.parametrize(
    "lags_and_window_size, serie, future_covariates, expected_y_train, expected_X_train",
    TEST_DATA_LAGS_CUSTOM_PREDICTORS,
)
def test_data_lags(
    lags_and_window_size,
    serie,
    future_covariates,
    expected_y_train,
    expected_X_train,
):

    lags, window_size = lags_and_window_size

    forecaster = RegressorForecastWithPredictors(
        regressor=LinearRegression(),
        lags=lags,
        freq=serie.index.freq,
        custom_predictors=np.mean,
        predictors_window_size=window_size,
    )
    X_train, y_train = forecaster.create_train_set(
        serie, future_covariates=future_covariates
    )

    assert X_train.eq(expected_X_train).all().all()
    assert X_train.index.equals(expected_X_train.index)
    assert y_train.eq(expected_y_train).all()
    assert y_train.index.equals(expected_y_train.index)
