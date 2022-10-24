import warnings
from itertools import cycle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression

# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.svm import OneClassSVM

from spark_forecast.models.regressor import RegressorForecast

warnings.filterwarnings("ignore")

# pylint: disable=unused-argument


SKLEARN_MODELS = [
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    GammaRegressor,
    HuberRegressor,
    LinearRegression,
    # LogisticRegression,
    PoissonRegressor,
    QuantileRegressor,
    TweedieRegressor,
    SVC,
    SVR,
    LinearSVC,
    OneClassSVM,
]


class CycleTestRegressor:
    def fit(self, X: pd.DataFrame, y_train: pd.Series) -> None:
        """Do nothing"""
        # We put this attribute because pytest doesn't allow
        # collect classes with __init__ constructor
        # pylint: disable=attribute-defined-outside-init
        self.dummy_returns = cycle([1, 2])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Just return 1 or 2"""
        return np.array([next(self.dummy_returns) for _ in range(X.shape[0])])


def dummy_models_factory(base_regressor):
    """We need this to test with all sckit-learn regressor with dummy predict"""

    class Test(base_regressor):
        def predict(self, X) -> np.ndarray:
            _ = super().predict(X)
            return np.ones(len(X))

    return Test


# TEST_DATA is a list of 5-tuples: 1 tp = regressor, 2 tp = tuple (lags, steps)
# 3 tp = time series, 4 tp = covariates (possible None), 5 tp = expected predictions
TEST_DATA_WITH_COV = [
    (
        dummy_models_factory(model)(),
        (3, 2),
        pd.Series(
            [10, 20, 30, 40, 50, 60],
            index=pd.date_range("2022-01-01", periods=6, freq="H"),
        ),
        pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6, 7, 8],
                "B": [True, True, True, True, False, False, False, False],
            },
            index=pd.date_range("2022-01-01", periods=8, freq="H"),
        ),
        pd.DataFrame(
            {
                "predictions": [1.0, 1.0],
                "lag_1": [60.0, 1.0],
                "lag_2": [50.0, 60.0],
                "lag_3": [40.0, 50.0],
                "A": [7.0, 8.0],
                "B": [0.0, 0.0],
            },
            index=pd.date_range("2022-01-01 06:00:00", periods=2, freq="H"),
        ),
    )
    for model in SKLEARN_MODELS
]

TEST_DATA_NONE_COV = [
    (
        dummy_models_factory(model)(),
        (3, 2),
        pd.Series(
            [10, 20, 30, 40, 50, 60],
            index=pd.date_range("2022-01-01", periods=6, freq="H"),
        ),
        None,
        pd.DataFrame(
            {
                "predictions": [1.0, 1.0],
                "lag_1": [60.0, 1.0],
                "lag_2": [50.0, 60.0],
                "lag_3": [40.0, 50.0],
            },
            index=pd.date_range("2022-01-01 06:00:00", periods=2, freq="H"),
        ),
    )
    for model in SKLEARN_MODELS
]


TEST_DATA_CYCLE = [
    (
        CycleTestRegressor(),
        (3, 2),
        pd.Series(
            [10, 20, 30, 40, 50, 60],
            index=pd.date_range("2022-01-01", periods=6, freq="H"),
        ),
        None,
        pd.DataFrame(
            {
                "predictions": [1.0, 2.0],
                "lag_1": [60.0, 1.0],
                "lag_2": [50.0, 60.0],
                "lag_3": [40.0, 50.0],
            },
            index=pd.date_range("2022-01-01 06:00:00", periods=2, freq="H"),
        ),
    ),
    (
        CycleTestRegressor(),
        (3, 2),
        pd.Series(
            [10, 20, 30, 40, 50, 60],
            index=pd.date_range("2022-01-01", periods=6, freq="H"),
        ),
        pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6, 7, 8],
                "B": [True, True, True, True, False, False, False, False],
            },
            index=pd.date_range("2022-01-01", periods=8, freq="H"),
        ),
        pd.DataFrame(
            {
                "predictions": [1.0, 2.0],
                "lag_1": [60.0, 1.0],
                "lag_2": [50.0, 60.0],
                "lag_3": [40.0, 50.0],
                "A": [7.0, 8.0],
                "B": [0.0, 0.0],
            },
            index=pd.date_range("2022-01-01 06:00:00", periods=2, freq="H"),
        ),
    ),
]
TEST_DATA_SPECIAL_ARGS = [
    (
        dummy_models_factory(KNeighborsRegressor)(n_neighbors=2),
        (3, 2),
        pd.Series(
            [10, 20, 30, 40, 50, 60],
            index=pd.date_range("2022-01-01", periods=6, freq="H"),
        ),
        None,
        pd.DataFrame(
            {
                "predictions": [1.0, 1.0],
                "lag_1": [60.0, 1.0],
                "lag_2": [50.0, 60.0],
                "lag_3": [40.0, 50.0],
            },
            index=pd.date_range("2022-01-01 06:00:00", periods=2, freq="H"),
        ),
    ),
    (
        dummy_models_factory(VotingRegressor)(
            (
                [
                    ("lr", LinearRegression()),
                    ("rf", AdaBoostRegressor()),
                    ("r3", RandomForestRegressor()),
                ]
            )
        ),
        (3, 2),
        pd.Series(
            [10, 20, 30, 40, 50, 60],
            index=pd.date_range("2022-01-01", periods=6, freq="H"),
        ),
        None,
        pd.DataFrame(
            {
                "predictions": [1.0, 1.0],
                "lag_1": [60.0, 1.0],
                "lag_2": [50.0, 60.0],
                "lag_3": [40.0, 50.0],
            },
            index=pd.date_range("2022-01-01 06:00:00", periods=2, freq="H"),
        ),
    ),
    (
        dummy_models_factory(KNeighborsRegressor)(n_neighbors=2),
        (3, 2),
        pd.Series(
            [10, 20, 30, 40, 50, 60],
            index=pd.date_range("2022-01-01", periods=6, freq="H"),
        ),
        pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6, 7, 8],
                "B": [True, True, True, True, False, False, False, False],
            },
            index=pd.date_range("2022-01-01", periods=8, freq="H"),
        ),
        pd.DataFrame(
            {
                "predictions": [1.0, 1.0],
                "lag_1": [60.0, 1.0],
                "lag_2": [50.0, 60.0],
                "lag_3": [40.0, 50.0],
                "A": [7.0, 8.0],
                "B": [0.0, 0.0],
            },
            index=pd.date_range("2022-01-01 06:00:00", periods=2, freq="H"),
        ),
    ),
]

TEST_DATA_ALL = (
    TEST_DATA_SPECIAL_ARGS
    + TEST_DATA_WITH_COV
    + TEST_DATA_NONE_COV
    + TEST_DATA_CYCLE
)


@pytest.mark.parametrize(
    "regressor, lags_and_steps, serie, future_covariates, expected_predict",
    TEST_DATA_ALL,
)
def test_fit_and_predict(
    regressor, lags_and_steps, serie, future_covariates, expected_predict
):

    forecaster = RegressorForecast(
        regressor=regressor,
        lags=lags_and_steps[0],
        freq=serie.index.freq,
    )

    if future_covariates is not None:
        forecaster.fit(
            serie, future_covariates=future_covariates[: len(serie)]
        )
        predictions = forecaster.predict(
            steps=lags_and_steps[1],
            future_covariates=future_covariates[len(serie) :],
        )
    else:
        forecaster.fit(serie)
        predictions = forecaster.predict(steps=lags_and_steps[1])

    assert predictions.eq(
        forecaster.observations_and_predictions["predictions"]
    ).all()
    assert (
        expected_predict.eq(forecaster.observations_and_predictions)
        .all()
        .all()
    )
    assert expected_predict.index.equals(
        forecaster.observations_and_predictions.index
    )
