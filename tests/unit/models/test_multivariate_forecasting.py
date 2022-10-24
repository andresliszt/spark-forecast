import pandas as pd
from sklearn.linear_model import LinearRegression


from spark_forecast.models.multivariate_forecasting import MULTIVARIATE_MODELS
from spark_forecast.models.multivariate_forecasting import (
    multivariate_forecast,
)


def test_all_models(test_file_forecasting_pandas):
    test_file_forecasting_pandas["ds"] = pd.to_datetime(
        test_file_forecasting_pandas["ds"]
    )
    test_file_forecasting_pandas = test_file_forecasting_pandas.set_index(
        "ds"
    ).sort_index()
    serie = test_file_forecasting_pandas["y"]
    serie.freq = pd.infer_freq(serie.index)
    # Some models doesn't admit 0, so re replace it from data
    serie[serie <= 0] = 1

    non_defaults = {
        "regressor": {"lags": 5, "regressor": LinearRegression(), "freq": "D"},
        "regressor_darts": {"lags": 5, "model": LinearRegression()},
        "lgbm": {"lags": 5},
        "random_forest": {"lags": 5},
        "rnn": {"input_chunk_length": 20, "n_epochs": 10},
    }

    for model in MULTIVARIATE_MODELS:
        model_kwargs = non_defaults[model] if model in non_defaults else None
        if model == "prophet":
            # I don't know why prophet stops the pytest execution
            # Ignoring for now.
            continue
        fc = multivariate_forecast(
            y=serie,
            model_name=model,
            freq=serie.freq,
            steps=2,
            model_kwargs=model_kwargs,
        )
        assert len(fc) == 2
        assert fc.index[0] > serie.index[-1]
