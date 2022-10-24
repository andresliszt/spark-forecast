import pandas as pd

from spark_forecast.models.local_forecasting import MODELS
from spark_forecast.models.local_forecasting import forecast


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
    for model in MODELS:
        fc = forecast(
            y=serie,
            model_name=model,
            freq=serie.index.freq,
            steps=2,
        )
        assert len(fc) == 2
        assert fc.index[0] > serie.index[-1]
