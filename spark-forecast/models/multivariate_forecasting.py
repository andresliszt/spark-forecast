# -*- coding: utf-8 -*-
"""Multivariate forecasting for application on Spark."""


from typing import Dict, Tuple
from typing import Optional
from typing import Union

import pandas as pd
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.auto_arima import AutoARIMA
from darts.models.forecasting.kalman_forecaster import KalmanForecaster
from darts.models.forecasting.sf_auto_arima import StatsForecastAutoARIMA
from darts.models.forecasting.random_forest import RandomForest
from darts.models.forecasting.gradient_boosted_model import LightGBMModel
from darts.models.forecasting.regression_model import RegressionModel
from darts.models.forecasting.rnn_model import RNNModel
from darts.timeseries import TimeSeries

try:
    from darts.models import Prophet
except ImportError:
    Prophet = None


from spark_forecast.exc import CovariatesTimeIndexDiffeError
from spark_forecast.exc import DateTimeStringError
from spark_forecast.exc import TooShortTimeSeries
from spark_forecast.exc import change_exception
from spark_forecast.models import logger
from spark_forecast.models.local_forecasting import PREDICTIONS_COL_NAME
from spark_forecast.models.local_forecasting import prepare_models_kwargs
from spark_forecast.models.regressor import RegressorForecast
from spark_forecast.preprocessing.utils import (
    check_is_datetime_indexed_and_frequency,
)
from spark_forecast.preprocessing.utils import check_time_index


FAIRY_MULTIVARIATE_MODELS = {
    "regressor": RegressorForecast,
}
"""All fary-forecast multivariate models supported."""

DARTS_MULTIVARIATE_MODELS = {
    "arima": ARIMA,
    "autoarima": AutoARIMA,
    "kalman": KalmanForecaster,
    "statsforecast_autoarima": StatsForecastAutoARIMA,
    "random_forest": RandomForest,
    "lgbm": LightGBMModel,
    "regressor_darts": RegressionModel,
    "rnn": RNNModel,
}
"""All darts multivariate models supported."""


if Prophet is not None:
    DARTS_MULTIVARIATE_MODELS["prophet"] = Prophet
else:
    logger.warning(
        "Facebook Prophet is not Installed. In order to use it"
        " you must install it inside your virtual environment. Isn't a mandatory dependency",
        installation_guide="https://facebook.github.io/prophet/docs/installation.html#installation-in-python",
    )

MULTIVARIATE_MODELS = {
    **FAIRY_MULTIVARIATE_MODELS,
    **DARTS_MULTIVARIATE_MODELS,
}
"""All multivariate models supported in this project."""


MULTIVARIATE_MODELS_INFO = {
    "regressor": {
        "docs": "Found on ``spark_forecast.models.regressor`` module",
        "probabilistic": True,
    },
    "regressor_darts": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression_model.html",
        "probabilistic": False,
    },
    "arima": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html",
        "probabilistic": True,
    },
    "autoarima": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.auto_arima.html",
        "probabilistic": False,
    },
    "kalman": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.kalman_forecaster.html",
        "probabilistic": True,
    },
    "statsforecast_autoarima": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_arima.html",
        "probabilistic": True,
    },
    "lgbm": {
        "docs": "Not docummented in Darts.",
        "probabilistic": True,
    },
    "random_forest": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.random_forest.html",
        "probabilistic": True,
    },
    "prophet": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.prophet_model.html",
        "probabilistic": True,
    },
    "rnn": {
        "docs": "https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html",
        "probabilistic": True,
    },
    "nbeats": {
        "docs": "https://unit8co.github.io/darts/examples/07-NBEATS-examples.html",
        "probabilistic": True,
    },
}


def fairy_fit_predict(
    y: pd.Series,
    model,
    steps: int,
    future_covariates_train: Optional[pd.DataFrame],
    future_covariates_predict: Optional[pd.DataFrame],
) -> pd.Series:
    """Fairy forecast fit-predict format.

    y
        Time series to be forecasted.
    model
        Model from dict `FAIRY_MULTIVARIATE_MODELS`
    steps
        Number of predictions.
    future_covariates_train
        Covariates for training.
    future_covariates_predict
        Covariates for predict.

    Returns
    -------
        Pandas Series of predictions of length
        equals to `steps`.

    """

    model.fit(y=y, future_covariates=future_covariates_train)
    # Finally we predict
    return model.predict(
        steps=steps, future_covariates=future_covariates_predict
    )


def __darts_covariates_to_timeseries(
    covariates: Optional[Union[pd.DataFrame, pd.Series]]
) -> Optional[TimeSeries]:
    """Transform to darts TimeSeries format."""
    if covariates is not None:
        return (
            TimeSeries.from_dataframe(covariates)
            if isinstance(covariates, pd.DataFrame)
            else TimeSeries.from_series(covariates)
        )
    return None


def darts_fit_predict(
    y: pd.Series,
    model,
    steps: int,
    future_covariates_train: Optional[Union[pd.Series, pd.DataFrame]],
    future_covariates_predict: Optional[Union[pd.Series, pd.DataFrame]],
):
    """Darts forecast fit-predict format.

    Basically, transform pandas Series/DataFrame
    to darts TimeSeries.

    y
        Time series to be forecasted.
    model
        Model from dict `DARTS_MULTIVARIATE_MODELS`.
    steps
        Number of predictions.
    future_covariates_train
        Covariates for training.
    future_covariates_predict
        Covariates for predict.

    Returns
    -------
        Pandas Series of predictions of length
        equals to `steps`.

    """

    model.fit(
        series=TimeSeries.from_series(y),
        future_covariates=__darts_covariates_to_timeseries(
            future_covariates_train
        ),
    )
    # Finally we predict
    return model.predict(
        n=steps,
        num_samples=1,
        future_covariates=__darts_covariates_to_timeseries(
            future_covariates_predict
        ),
    ).pd_series()


def check_covariates(
    future_covariates: pd.DataFrame,
    freq: str,
    steps: int,
    start_time: str,
    current_time: str,
) -> None:
    # We check time index and frequency
    check_time_index(future_covariates.index, freq)
    # We check also string format of datetimes
    with change_exception(
        DateTimeStringError(
            string_datetimes=[start_time, current_time], freq=freq
        ),
        ValueError,
    ):
        # Whole history needed for do a forecast
        covariates_train_time = pd.date_range(
            start_time,
            periods=len(pd.date_range(start_time, current_time, freq=freq)),
            freq=freq,
        )
        covariates_predict_time = pd.date_range(
            current_time, periods=steps + 1
        )[1:]

    # We check if covariates contains th whole history
    with change_exception(
        CovariatesTimeIndexDiffeError(
            start_time=start_time,
            current_time=current_time,
            steps=steps,
            freq=freq,
        ),
        KeyError,
    ):
        return (
            future_covariates.loc[covariates_train_time],
            future_covariates.loc[covariates_predict_time],
        )


def prepare_multivariate_forecast(
    y: pd.Series,
    freq: str,
    steps: int = 1,
    future_covariates: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:

    # We verify that y is a suitable time serie
    check_is_datetime_indexed_and_frequency(y, freq=freq)
    if future_covariates is not None:
        # Note that the checker ensure us that the serie is perfectly sorted
        start_time = y.index[0]
        # current_time in this method is the last observation
        current_time = y.index[-1]
        # We get the covariates needed to do forecast
        future_covariates_train, future_covariates_predict = check_covariates(
            future_covariates,
            freq=freq,
            steps=steps,
            start_time=start_time,
            current_time=current_time,
        )
        return future_covariates_train, future_covariates_predict
    return None, None


def multivariate_forecast(
    y: pd.Series,
    model_name: str,
    freq: str,
    steps: int,
    future_covariates: Optional[pd.DataFrame] = None,
    model_kwargs: Optional[Dict] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Simple call to univariate forecasts from ``darts``.

    ``y`` must be a suitable time series with correct time index and
    defined frequency.

    """

    model_kwargs = prepare_models_kwargs(model_kwargs)

    (
        future_covariates_train,
        future_covariates_predict,
    ) = prepare_multivariate_forecast(
        y=y,
        freq=freq,
        steps=steps,
        future_covariates=future_covariates,
    )

    if model_name in FAIRY_MULTIVARIATE_MODELS:
        model = FAIRY_MULTIVARIATE_MODELS[model_name](**model_kwargs)
        model.fit(y=y, future_covariates=future_covariates_train)
        pred = model.predict(
            steps=steps, future_covariates=future_covariates_predict
        )

    elif model_name in DARTS_MULTIVARIATE_MODELS:
        model = DARTS_MULTIVARIATE_MODELS[model_name](**model_kwargs)
        model.fit(
            series=TimeSeries.from_series(y),
            future_covariates=__darts_covariates_to_timeseries(
                future_covariates_train
            ),
        )
        pred = model.predict(
            n=steps,
            num_samples=1,
            future_covariates=__darts_covariates_to_timeseries(
                future_covariates_predict
            ),
        ).pd_series()

    else:
        raise ValueError(
            "Invalid model name. Must be one of "
            f"{[*list(FAIRY_MULTIVARIATE_MODELS),*list(DARTS_MULTIVARIATE_MODELS)]}"
        )
    if model.min_train_series_length > len(y):
        raise TooShortTimeSeries(
            length_received=len(y),
            length_minimum=model.min_train_series_length,
        )
    # Name that will be used in Spark.
    pred.name = PREDICTIONS_COL_NAME

    return pred


def probabilistic_multivariate_forecast(
    y: pd.Series,
    model_name: str,
    freq: str,
    steps: int,
    num_samples: int,
    future_covariates: Optional[pd.DataFrame] = None,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
    model_kwargs: Optional[Dict] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Simple call to multivariate forecasts from ``darts``.

    ``y`` must be a suitable time series with correct time index and
    defined frequency.

    """
    model_kwargs = prepare_models_kwargs(model_kwargs)

    (
        future_covariates_train,
        future_covariates_predict,
    ) = prepare_multivariate_forecast(
        y=y,
        freq=freq,
        steps=steps,
        future_covariates=future_covariates,
    )

    # Single prediction first
    if model_name in FAIRY_MULTIVARIATE_MODELS:
        model = FAIRY_MULTIVARIATE_MODELS[model_name](**model_kwargs)
        model.fit(y=y, future_covariates=future_covariates_train)
        # Fairy probabilistic interval returns predictions too
        return model.predict_interval(
            y=y,
            steps=steps,
            n_bootstraps=num_samples,
            future_covariates=future_covariates_predict,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    if model_name in DARTS_MULTIVARIATE_MODELS:
        model = DARTS_MULTIVARIATE_MODELS[model_name](**model_kwargs)
        model.fit(
            series=TimeSeries.from_series(y),
            future_covariates=__darts_covariates_to_timeseries(
                future_covariates_train
            ),
        )
        # We get single pred
        future_covariates_predict = __darts_covariates_to_timeseries(
            future_covariates_predict
        )

        single_pred = model.predict(
            n=steps, num_samples=1, future_covariates=future_covariates_predict
        ).pd_series()
        # We get darts probabilistic interval
        probabilistic_pred = model.predict(
            n=steps,
            num_samples=num_samples,
            future_covariates=future_covariates_predict,
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

    raise ValueError(
        "Invalid model name. Must be one of "
        f"{[*list(FAIRY_MULTIVARIATE_MODELS),*list(DARTS_MULTIVARIATE_MODELS)]}"
    )
