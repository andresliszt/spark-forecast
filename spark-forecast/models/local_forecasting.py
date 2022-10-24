# -*- coding: utf-8 -*-
"""Local Forecasting models from ``darts``."""

from typing import Dict
from typing import Optional
from typing import Union

import pandas as pd
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.auto_arima import AutoARIMA
from darts.models.forecasting.croston import Croston
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.theta import FourTheta
from darts.models.forecasting.theta import Theta
from darts.timeseries import TimeSeries

from spark_forecast.exc import TooShortTimeSeries
from spark_forecast.preprocessing.utils import (
    check_is_datetime_indexed_and_frequency,
)

MODELS = {
    "fast_fourier_transform": FFT,
    "exponential_smoothing": ExponentialSmoothing,
    "arima_univariate": ARIMA,
    "four_theta": FourTheta,
    "theta": Theta,
    "croston": Croston,
    "autoarima_univariate": AutoARIMA,
}
"""All univariate models."""

MODELS_INFO = {
    "fast_fourier_transform": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.fft.html",
        "probabilistic": False,
    },
    "exponential_smoothing": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.exponential_smoothing.html",
        "probabilistic": True,
    },
    "arima_univariate": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html",
        "probabilistic": True,
    },
    "four_theta": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html",
        "probabilistic": False,
    },
    "theta": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html",
        "probabilistic": False,
    },
    "croston": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.croston.html",
        "probabilistic": False,
    },
    "autoarima_univariate": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.auto_arima.html",
        "probabilistic": False,
    },
    "tbats": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tbats.html",
        "probabilistic": True,
    },
    "bats": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tbats.html",
        "probabilistic": True,
    },
}

PREDICTIONS_COL_NAME = "predictions"
"""Col name for the output."""


def prepare_models_kwargs(model_kwargs: Optional[Dict]) -> Optional[Dict]:
    return {} if model_kwargs is None else model_kwargs


def forecast(
    y: pd.Series,
    model_name: str,
    freq: str,
    steps: int = 1,
    model_kwargs: Optional[Dict] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Simple call to univariate forecasts from ``darts``.

    ``y`` must be a suitable time series with correct time index and
    defined frequency.

    """
    # If not dict is provided, then we use defaults of each models
    model_kwargs = prepare_models_kwargs(model_kwargs)
    # We verify that y is a suitable time serie
    check_is_datetime_indexed_and_frequency(y, freq=freq)
    # We get an available model
    model = MODELS[model_name](**model_kwargs)
    # Models doesn't support too short series
    if model.min_train_series_length > len(y):
        raise TooShortTimeSeries(
            length_received=len(y),
            length_minimum=model.min_train_series_length,
        )
    # We fit the model
    model.fit(series=TimeSeries.from_series(y))
    # Finally we predict
    pred = model.predict(n=steps, num_samples=1).pd_series()
    # The name will be used on Spark API
    pred.name = PREDICTIONS_COL_NAME

    return pred


def probabilistic_forecast(
    y: pd.Series,
    model_name: str,
    freq: str,
    steps: int = 1,
    num_samples=1,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
    model_kwargs: Optional[Dict] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Simple call to univariate forecasts from ``darts``.

    ``y`` must be a suitable time series with correct time index and
    defined frequency.

    """

    # We verify that y is a suitable time serie
    check_is_datetime_indexed_and_frequency(y, freq=freq)
    model_kwargs = prepare_models_kwargs(model_kwargs)
    model = MODELS[model_name](**model_kwargs)

    if model.min_train_series_length > len(y):
        raise TooShortTimeSeries(
            length_received=len(y),
            length_minimum=model.min_train_series_length,
        )

    # We transform to TimeSeries Object
    y_darts = TimeSeries.from_series(y)
    # We fit the model
    model.fit(series=y_darts)
    # Single prediction first
    single_pred: TimeSeries = model.predict(n=steps, num_samples=1).pd_series()
    single_pred.name = PREDICTIONS_COL_NAME
    # We will use it for probabilistic forecast
    probabilistic_pred: TimeSeries = model.predict(
        n=steps, num_samples=num_samples
    )
    # We use built-in darts methods calculate quantiles
    pred_probabilistic = pd.concat(
        [
            single_pred,
            probabilistic_pred.quantile_df(quantile=lower_quantile),
            probabilistic_pred.quantile_df(quantile=upper_quantile),
        ],
        axis=1,
    )
    pred_probabilistic.columns = [
        PREDICTIONS_COL_NAME,
        f"lower_quantile_{lower_quantile}",
        f"upper_quantile_{upper_quantile}",
    ]
    return pred_probabilistic
