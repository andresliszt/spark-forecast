# Fairy forecast

This repository aims to implement [darts](https://unit8co.github.io/darts/) on top of Spark trough Pyspark.

# Installation

This project uses:

-   Python 3.9.x
-   [darts](https://unit8co.github.io/darts/)
-   [Prophet](https://facebook.github.io/prophet/) This is an optional dependency and if needed must be installed using ``pip``. Prophet has its own requirements.
-   [Poetry](https://python-poetry.org/) Only needed for installation from git repository. Installing wheels from private PyPI doesn't require it.


## Cloning from git repository

```sh
    git https://github.com/andresliszt/spark-forecast
    cd spark-forecast
    python -m venv .venv
    poetry install
```


# Usage

This project implements several time series preprocessing

## Time Series cleansing


```python

from spark_forecast.helpers._pyspark.time_series import SparkTimeSeriesHelper

timeseries_helper = SparkTimeSeriesHelper(
    target_column="y",
    time_column="ds",
    group_columns=["location_id", "item_id"],
    freq="D",
)

```

Here ``target_column`` is the column of ``data`` that refers to the target that we want to clean/forecast, ``time_column`` is the time column, ``group_columns`` can be a single column (``str``) or multiples columns (``List[str]``) that uniquely identify time series in ``data``, and note that it's a very important detail, because some product may have the same code among supermarkets. Finally ``freq`` is the frequency of all time series in ``data`` and must be a valid ``pandas`` [offset](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).

In order the clean all time series in ``data``


```python

from spark_forecast.helpers._pyspark.time_series import SparkTimeSeriesHelper

data_spark_daily = timeseries_helper.fill_and_fix_time_series_spark(
    data=data,
    duplicate_strategy="sum",
    interpolator={"method": "linear", "limit": 10},
    fillna_strategy="mean",
    outliers_detection={"method": "iqr"},
    zero_bfill_to="2021-01-01",
    repeat_ffill_to="2022-10-01",
)



```

``duplicate_strategy`` referes to the aggregation for the case if two dates are repeated for a single product, and valid values are ``mean``, ``mode``, ``median``, ``sum``. The ``interpolator`` is a dictionary passed to the pandas [interpolator](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html) method and note that some keyword are invalid in this context and ignored therefore (for example passing ``inplace = True`` is ignored.). The interpolation is the first action barrier to combat NaNs values and may not fill all those values because it depends on the ``method``, for example ``interpolator={"method": "linear", "limit": 10}`` will fill only a maximum number of 10 consecutive NaNs. The ``interpolator`` keyword is *optional*. After the optional interpolation ``fillna_strategy`` is a mandatory keyword that ensure us that all NaNs will be filled (if interpolation is passed, it will fill only NaNs the couldn't be filled), and valid options are ``mean``, ``zero`` or ``median``. The ``outliers_detection`` is a dictionary whose ``method`` key referes to one the ``spark_forecast.outlier_detection.outlier_detection.OUTLIERS_METHODS`` and the next keys referes keyword arguments for the method, and for the moment outliers are replaced with NaNs and filled after with the interpolator and NaN filler process. The ``zero_bfill_to`` is a date string for which all time series will be back extended with zeros, and ignored if the time serie has older records than this date, and also ``repeat_bfill_to`` is a date string for which all time series will be backward repeated. Also exists the ``repeat_ffill_to`` that do a forward fill with the last value found. Note that ``zero_bfill_to``, ``repeat_bfill_to`` and ``repeat_ffill_to`` are optional and we don't recommend, but are important in the context of hierarchical forecast, where all time series must have the same time horizont.

> **Warning**
> If any time series is completely null, ``fairy-forecast`` will raise a error, we recommend to drop those series before going into Spark.


## Time Series classification

The [paper](https://link.springer.com/article/10.1007/s12351-020-00605-2) *Comparison of statistical and machine learning methods for daily SKU demand forecasting* classifies the time series into five categories *zero*, *lumpy*, *erratic*, *Intermitent* and *smooth*. We have implemented these categorizations in ``SparkTimeSeriesHelper``.


```python

classifications = time_helpers.naive_time_series_classifier_spark(data = data).cache()

```

The unique values given by ``naive_time_series_classifier_spark`` method are ``Smooth``, ``Erratic``, ``Zero``, ``Lumpy`` and ``Intermitent`` and ``classifications`` may not contain all these classifications.

> **IMPORTANT**
> Note that these classifications should be done prior data cleansing and can be applied post cleansing to see how classifications changes.


## Time Series cleansing based on classifications

Given a group of classifications, for example, as showed above, we can apply a different data cleansing for each classification. We only need another Spark DataFrame containing a classification for each unique time series found in ``data``, that is, a classification per ``group_columns`` unique items. Using the above example with ``classifications``, we define a dictionary ``ARGS_PER_CLASSIFICATION`` with keys given the values of ``classifications`` and the values are dicctionary passed to the inner method of ``fill_and_fix_time_series_spark`` for cleansing.


```python

ARGS_PER_CLASSIFICATION = {
    "Smooth": {
        "fillna_strategy": "mean",
        "duplicate_strategy": "sum",
        "interpolator": {"method": "linear"},
        "outliers_detection": {"method": "iqr"},
    },
    "Zero": {
        "fillna_strategy": "zero",
        "duplicate_strategy": "sum",
    },
    "Erratic": {
        "fillna_strategy": "zero",
        "duplicate_strategy": "sum",

    },
    "Lumpy": {
        "fillna_strategy": "zero",
        "duplicate_strategy": "sum",
    },
    "Intermitent": {
        "fillna_strategy": "zero",
        "duplicate_strategy": "sum",
    },
}


data_spark_daily = (
    time_helpers.fill_and_fix_time_series_spark_based_on_classification(
        data = data,
        classification_data = classifications,
        fill_dict=ARGS_PER_CLASSIFICATION,
        zero_bfill_to="2021-01-01",
        repeat_ffill_to="2022-10-01",
    ).cache()
)

```

> **REMARK**
> Is user responsability ensure that ``data`` and ``classification_data`` given in the ``fill_and_fix_time_series_spark_based_on_classification``
> method match in the join operation, this includes ensure that join columns ``group_columns`` must be same data type. You can use any
>  ``classification_data`` to do this not only the given by the method ``naive_time_series_classifier_spark``.



## Weekly aggregation

This is a helper that builds weekly data from daily. At this point **the data must be with perfect daily time column**, otherwise custom errors will be raised, so we recommend use this method after of calling ``fill_and_fix_time_series_spark`` or ``fill_and_fix_time_series_spark_based_on_classification``. We need also at least 1 week of observation for all series in order to use it, otherwise error will be raised (note how ridiculous sounds this, but in the raw data sometimes exists series with less than 7 records.)



```python

data_weekly = time_helpers.build_weekly_aggregation_spark(data = data_spark_daily, day = "MON").cache()

```

The method recives ``day`` parameter (default ``day = MON``) and represents the start day in the week. Valid options are ``MON``, ``TUE``, ``WED``, ``THU``, ``FRI``, ``SAT`` and ``SUN``.

> **WARNING**
> Be sure that all time series in the data can be **perfectly aggregated in weeks**, that is, given the ``day`` parameter the first day and the last day
> in the time horizont for each time series should be the choosen ``day``, otherwise the last week (and the first) could be aggregated with less days
> being a problem for the forecasting. You could use ``repeat_ffill_to``, ``zero_bfill_to``, or ``repeat_bfill_to`` arguments of the cleansing methods
> to handle this. This is just a warning, and the method will run anyway.


## Forecasting

We have so far univariate and multivariate forecasting.

We have two error handling

* ``fairy-forecast`` will extract the first time series found in the data, and will try to do a single forecast with that serie, and if fails will raise the error. This will discard all *dummy* errors as bad keywords arguments passed to the models, and also could detect if the data going into the forecast is in a bad format, for example with null values or duplicated dates. Note that this is only done in the first time series, but is very probably that if the first one is in a bad format, the others will be too, so you may check your data and check if you used well the cleansing methods.

* If the validation on the first series worked, then all errors found in the next series will be ignored and placed in the columns ``error`` and ``type_error`` of the returning forecast Data Frame. This is because some models could trough errors for some time series (some models doesn't admit zeros) and we don't want a fail after a long runing execution.


### Univariate Forecasting


The univariate forecast is the simpliest and the available methods can be found by


```python

from spark_forecast.helpers._pyspark.forecast import univariate_models_info

>>> univariate_models_info()

{'fast_fourier_transform': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.fft.html',
  'probabilistic': False},
 'exponential_smoothing': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.exponential_smoothing.html',
  'probabilistic': True},
 'arima_univariate': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html',
  'probabilistic': True},
 'four_theta': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html',
  'probabilistic': False},
 'theta': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html',
  'probabilistic': False},
 'croston': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.croston.html',
  'probabilistic': False},
 'autoarima_univariate': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.auto_arima.html',
  'probabilistic': False},
 'tbats': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tbats.html',
  'probabilistic': True},
 'bats': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tbats.html',
  'probabilistic': True}}


```

If you want apply the same univariate model for all time series found on ``data``

```python

from spark_forecast.helpers._pyspark.forecast import forecast_spark


forecast = forecast_spark(
    data=data_spark_daily,
    freq = "D",
    target_column="y",
    time_column="ds",
    group_columns=["location_id", "item_id"],
    model_name="exponential_smoothing",
    steps=2,
).cache()


```
The method by default predicts so many time in future as ``steps`` parameter and **the starter point of the predictions depends on each time series**, for example suppose a time series found in ``data`` has time horizont to "2022-01-01", then the predictions (with ``steps = 2``) will be for "2022-01-02" and
"2022-01-03", and if other has time horizont to "2022-01-10", then the predictions will be "2022-01-11" and "2022-01-12". If you need that all time series to have the same forecast horizont you must ensure that all time series in ``data_spark_daily`` have the same time horizont, that can be done using ``repeat_ffill_to`` of the cleansing methods.


### Probabilistic Univariate Forecasting

We do probaiblistic forecast and we return interval of predictions. All probabilistic models have a different way to do this and we recommend to read the documentation. The probabilistic univariate models are listed in ``univariate_models_info`` function.

```python

from spark_forecast.helpers._pyspark.forecast import probabilistic_forecast_spark


forecast = probabilistic_forecast_spark(
    data=data_spark_daily,
    target_column="y",
    freq = "D",
    time_column="ds",
    group_columns=["location_id", "item_id"],
    model_name="exponential_smoothing",
    upper_quantile = 0.75,
    lower_quantile = 0.25,
    steps=2,
).cache()

```

``num_samples`` is the quantity of forecasting point will be generated (i.e the distribution) and ``lower_quantile``and ``upper_quantile`` are the quantiles that we will return of the distribution.


### Multivariate Forecasting

Same logic as the [univariate forecasting](#univariate-forecasting) but here we introduce the concept of *future_covariates*, which are exogeneous variables known in the whole
time horizont, i.e for training and for predict. For the moment these covariates are a ``pandas`` Data Frame that will work as exogeneous variables for all time series found in the Spark DataFrame. For example

The documentation of the models can be obtained by doing

```python

from spark_forecast.helpers._pyspark.forecast import multivariate_models_info

>>> multivariate_model_info()

{'regressor': {'docs': 'Found on ``spark_forecast.models.regressor`` module',
  'probabilistic': True},
 'regressor_darts': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression_model.html',
  'probabilistic': False},
 'arima': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html',
  'probabilistic': True},
 'autoarima': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.auto_arima.html',
  'probabilistic': False},
 'kalman': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.kalman_forecaster.html',
  'probabilistic': True},
 'statsforecast_autoarima': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_arima.html',
  'probabilistic': True},
 'lgbm': {'docs': 'Not docummented in Darts.', 'probabilistic': True},
 'random_forest': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.random_forest.html',
  'probabilistic': True},
 'prophet': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.prophet_model.html',
  'probabilistic': True},
 'rnn': {'docs': 'https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html',
  'probabilistic': True},
 'nbeats': {'docs': 'https://unit8co.github.io/darts/examples/07-NBEATS-examples.html',
  'probabilistic': True}}


```

So we do a multivariate forecasting

```python

import numpy as np
import pandas a pd
from sklearn.ensemble import RandomForestRegressor

from spark_forecast.helpers._pyspark.forecast import multivariate_forecast_spark

time_index = pd.date_range("2021-01-01", "2022-10-03")

dummy_covariates = pd.DataFrame(
    {
        "A": np.random.randint(0,2, size = len(time_index)),
        "B": np.random.randint(0,2, size = len(time_index))
    },
    index = time_index
)

forecast = multivariate_forecast_spark(
    data=data_spark_daily,
    target_column="y",
    freq = "D",
    time_column="ds",
    group_columns=["location_id", "item_id"],
    model_name="regressor",
    model_kwargs = {"lags": 30, "regressor": RandomForestRegressor(max_depth = 100, n_estimators = 200)},
    future_covariates = dummy_covariates,
    steps=2,
).cache()


```

All time series found in the Spark Data Frame must satisfy that time horizont plus the forecast horizont must becontained in the ``future_covariates`` time index. Note careful that the ``future_covariates`` index is an instance of ``pandas.core.indexes.datetimes.DatetimeIndex``, otherwise errors will be raised.


```python

import numpy as np
import pandas a pd

from spark_forecast.helpers._pyspark.forecast import multivariate_forecast_spark

time_index = pd.date_range("2021-01-01", "2022-10-03")

dummy_covariates = pd.DataFrame(
    {
        "A": np.random.randint(0,2, size = len(time_index)),
        "B": np.random.randint(0,2, size = len(time_index))
    },
    index = time_index
)

forecast = multivariate_forecast_spark(
    data=data_spark_daily,
    target_column="y",
    freq = "D",
    time_column="ds",
    group_columns=["location_id", "item_id"],
    model_name="prophet",
    future_covariates = dummy_covariates,
    steps=2,
).cache()


```

An example using prophet, note that in this case ``model_kwargs`` can be omited because it has default values for ``prophet`` model, but if you want to give diferent than default you may use ``model_kwargs`` after read the documentation of the model.


> **REMARK**
> The ``multivariate_forecast_spark`` can be used without ``future_covariates`` argument, i.e is optional, because all the multivariate models supports
> forecast without exogeneous variables.


### Probabilistic Multivariate Forecasting

Same as the multivariate forecasting presented above and the probabilistic univariate forecasting.

```python

import numpy as np
import pandas a pd

from spark_forecast.helpers._pyspark.forecast import probabilistic_multivariate_forecast_spark

time_index = pd.date_range("2021-01-01", "2022-10-03")

dummy_covariates = pd.DataFrame(
    {
        "A": np.random.randint(0,2, size = len(time_index)),
        "B": np.random.randint(0,2, size = len(time_index))
    },
    index = time_index
)

forecast = probabilistic_multivariate_forecast_spark(
    data=data_spark_daily,
    target_column="y",
    freq = "D",
    time_column="ds",
    group_columns=["location_id", "item_id"],
    model_name="prophet",
    future_covariates = dummy_covariates,
    lower_quantile = 0.25,
    upper_quantile = 0.75,
    num_samples = 20,
    steps=2,
).cache()


```

### Forecasting based on classifications


Same as the cleansing based on classification, if we have another Spark Data Frame with classifications for each time series, we can do a different forecast for each classification. We use ``classifications`` defined in the [Classification Section](#time-series-classification).

```python

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


from spark_forecast.helpers._pyspark.forecast import (
    multivariate_forecast_spark,
    forecast_with_classification_spark,
)

MODEL_DICT = {
    "Smooth": {
        "model_name": "regressor",
        "lags": 4,
        "regressor": RandomForestRegressor(max_depth=40),
    },
    "Zero": {"model_name": "exponential_smoothing"},
    "Erratic": {
        "model_name": "regressor",
        "lags": 6,
        "regressor": XGBRegressor(),
    },
    "Lumpy": {"model_name": "exponential_smoothing"},
    "Intermittent": {"model_name": "fast_fourier_transform"},
}

forecast = forecast_with_classification_spark(
    data=data_spark_daily,
    freq="D",
    steps=2,
    group_columns=["location_id", "item_id"],
    model_classification_dict=MODEL_DICT,
    classification_data=classifications,
    target_column="y",
    time_column="ds",
).cache()


```

So, here ``model_classification_dict`` is a dictionary whose keys are the unique values of Spark ``classifications`` passed trough ``classification_data`` argument. Each inner dictionary must contain the key ``model_name`` specifying a valid model name supported in this project, the rest of the keys are the keywords arguments passed to the model inizialization, and as always, some models have defaults values, and probably mostly ``model_name`` will be the only mandatory key.


## Developing

To create new features and improve the project you must set a development enviroment, because you need to use Spark. There are two needed things to take in mind. First, you will need an enviroment where you can test your code in programming time, and a jupyter notebook is a very nice option. We recommend to use the [pyspark-notebook](https://hub.docker.com/r/jupyter/pyspark-notebook) and mount a volume of the code folder ``spark_forecast`` inside a container of that image (here ``spark_forecast`` is the inner folder where are placed all ``.py`` files). The volume will help to update all the fairy forecast files inside the container at programming time.

### Testing

For testing, we don't need ``pyspark-notebook``, because that is a development tool. We use a minimal Docker image with a Spark environment in order to run the integration test, this image is build with the ``Dockerfile`` of the project. The test will run with the Azure Pipeline as part of the CI process. We also recommend to create a local container with a mounted volume in order to run the test local before going into the pipeline. To do that we do the following commands

```sh
docker build -t fairy_tester_image --target tester .

```
This command will create an image that allow us to create containers with ``fairy-forecast`` ``.py`` files and the ``tests`` folder. Next you need to create a continaer with a mounted volume to update the ``.py`` files at programming time

```sh

docker run -v <FULL_PATH_TO_spark_forecast_PY_FILES_FOLDER>:/spark_forecast/spark_forecast -v <FULL_PATH_TO_spark_forecast_TEST_FOLDER>:/spark_forecast/tests  --name fairy_test_container fairy_tester_image

```

The command above will create a container called ``fairy_test_container`` with a volume between our local ``.py`` files and the container.


## TODO

* Parameter tunning
* Darts ``past_covariates``
