# -*- coding: utf-8 -*-
"""Forcast execution on spark."""

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import pyspark
from pyspark.sql.types import DateType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType


from spark_forecast.exc import ColumnNamesError
from spark_forecast.exc import ModelInitializationError
from spark_forecast.exc import change_exception
from spark_forecast.exc import NotProbabilisticModelError
from spark_forecast.helpers._pyspark.utils import group_columns_to_list
from spark_forecast.helpers._pyspark.utils import (
    return_dataframe_for_spark_api,
)
from spark_forecast.helpers._pyspark.utils import (
    return_dataframe_probabilistic_for_spark_api,
)
from spark_forecast.helpers._pyspark.utils import validate_classification_dict
from spark_forecast.models.local_forecasting import MODELS
from spark_forecast.models.local_forecasting import MODELS_INFO
from spark_forecast.models.local_forecasting import forecast
from spark_forecast.models.local_forecasting import probabilistic_forecast
from spark_forecast.models.multivariate_forecasting import MULTIVARIATE_MODELS
from spark_forecast.models.multivariate_forecasting import (
    MULTIVARIATE_MODELS_INFO,
    FAIRY_MULTIVARIATE_MODELS,
)
from spark_forecast.models.multivariate_forecasting import (
    multivariate_forecast,
    probabilistic_multivariate_forecast,
)


def __fairy_default_freq(model_name, model_kwargs, freq):
    if model_name in FAIRY_MULTIVARIATE_MODELS:
        model_kwargs.setdefault("freq", freq)
    return model_kwargs


def __model_info(model_dict: Dict[str, str]):
    try:
        from IPython.display import display

        display(model_dict)
    except ImportError:
        print(model_dict)


def univariate_model_info():
    __model_info(model_dict=MODELS_INFO)


def multivariate_model_info():
    __model_info(model_dict=MULTIVARIATE_MODELS_INFO)


def set_time_index(data: pd.DataFrame, time_column: str):
    data[time_column] = pd.to_datetime(data[time_column])
    # We set time index
    return data.set_index(time_column).sort_index()


def __extract_first_serie(
    data: pyspark.sql.dataframe.DataFrame,
    group_columns: List[str],
    target_column: str,
    time_column: str,
) -> pd.Series:

    """Extract first time series from ``data``

    Used to check if a given model runs on this
    single series, in order to detect *dummy errors*

    Returns
    -------
        A time series to be tested.
    """

    first_row = data.first()
    # We get only
    group_columns_values = {
        col: getattr(first_row, col) for col in group_columns
    }

    condition = None

    for col, val in group_columns_values.items():
        # We filter by each group column
        if condition is None:
            condition = data[col] == val
        else:
            condition = condition & (data[col] == val)

    serie = data[condition]

    return set_time_index(serie.toPandas(), time_column)[target_column]


def __check_if_can_be_initialized(
    model_name: str,
    model_kwargs: Optional[Dict],
    check_probabilistic: bool = False,
) -> None:
    """Analyzes that model can be initialized before going into Spark.

    Useful because all forecast going to Spark are wrapped with a
    try/except clause, and we don't want inizialization errors.

    """

    model_kwargs = {} if model_kwargs is None else model_kwargs

    models = {**MODELS, **MULTIVARIATE_MODELS}

    with change_exception(
        ValueError(
            f"Available models for forecast are {list(models.keys())}. Got {model_name}"
        ),
        KeyError,
    ):
        model = models[model_name]

    with change_exception(
        ModelInitializationError(
            type_model=str(model), arguments=model_kwargs
        ),
        BaseException,
    ):
        model_obj = model(**model_kwargs)

    if check_probabilistic:
        if not model_obj._is_probabilistic():
            raise NotProbabilisticModelError(type_model=type(model))


def __validate_model_dict_with_classification(
    classification_data: pyspark.sql.dataframe.DataFrame,
    model_classification_dict: Dict[str, Dict],
    freq: str,
) -> None:
    """Analizes if all models in ``model_classification_dict`` can be initialized.

    Here ``classification_data`` is a pyspark DataFrame containing
    a column called ``classification`` and keys columns to identify
    uniquely each time series. The dictionary ``model_classification_dict``
    must has a key for each unique classification found in the
    column ``classification`` and the value for each key
    is another dictionary containing a key called ``model_name``
    with a valid name for a ``spark_forecast`` model, and the
    remainder keys are the keyword arguments passed to the model.

    This function is hidden and used before going into Spark in
    :func:`forecast_with_classification_spark`


    Parameters
    ----------
    classification_data
        Spark DataFrame with classification for each time series.
    model_classification_dict
        Dictionary specifying a model for each unique classification
        found in ``classification_data``.

    Raises
    ------
    ValueError
        If a malformed dictionary is passed.
    ModelInitializationError
        If any of the models given in ``model_classification_dict``
        can't be initialized.

    """

    validate_classification_dict(
        classification_data, model_classification_dict
    )
    for model_kwargs in model_classification_dict.values():
        if "model_name" not in model_kwargs:
            raise ValueError(
                "Inner dict must contain `model_name` key with a valid `spark_forecast` model"
            )
        __fairy_default_freq(
            model_kwargs["model_name"], model_kwargs, freq=freq
        )
        _model_kwargs = model_kwargs.copy()
        __check_if_can_be_initialized(
            _model_kwargs.pop("model_name"), _model_kwargs
        )


def __validate_common_params(
    steps: int,
    time_column: str,
    target_column: str,
    group_columns: List[str],
    real_columns: List[str],
    lower_quantile=None,
    upper_quantile=None,
) -> None:
    """Validates common parameters before going into Spark."""

    if not isinstance(steps, int) or steps < 1:
        raise ValueError(
            "``steps`` parametr must be integer greater or equal than 1."
        )

    if not {target_column, time_column, *set(group_columns)} <= set(
        real_columns
    ):
        raise ColumnNamesError(
            columns=real_columns,
            requested_columns=list(
                [target_column, time_column, *group_columns]
            ),
        )
    if upper_quantile is not None:
        if not 0 <= lower_quantile <= 1 or not 0 <= upper_quantile <= 1:
            raise ValueError(
                "`lower_quantile` and `upper_quantile` must be float between 0 and 1 (inclusive)"
            )


def __build_common_schema(
    time_column: str,
    group_columns: List[str],
) -> List[StructField]:
    """Common schema for returning DataFrame.

    These are common field for forecast and probabilistic forecast that
    will be returned in the applyInPandas method.

    """

    return [
        StructField("predictions", DoubleType(), nullable=True),
        StructField(time_column, DateType(), nullable=True),
        *[
            StructField(col, StringType(), nullable=False)
            for col in group_columns
        ],
        StructField("error", StringType(), nullable=True),
        StructField("type_error", StringType(), nullable=True),
    ]


def __build_forecast_schema(
    time_column: str, group_columns: List[str]
) -> StructType:
    """Forecast schema for Spark."""
    return StructType(__build_common_schema(time_column, group_columns))


def __build_forecast_probabilistic_schema(
    time_column: str,
    group_columns: List[str],
    lower_quantile: float,
    upper_quantile: float,
) -> StructType:
    """Probabilistic forecast schema for Spark."""
    return StructType(
        [
            *__build_common_schema(time_column, group_columns),
            StructField(
                f"lower_quantile_{lower_quantile}", DoubleType(), nullable=True
            ),
            StructField(
                f"upper_quantile_{upper_quantile}", DoubleType(), nullable=True
            ),
        ]
    )


def _prepare_grouped_data(
    data: pd.DataFrame,
    target_column: str,
    time_column: str,
    group_columns: List[str],
) -> Tuple[pd.Series, Dict[str, str]]:
    """Prepare time series before forecasting.

    This is a private function to prepare the time series that can be
    extracted from ``data``, that is a grouped pandas Data Frame comming
    from a group by operation and an applyInPandas function. This method
    sort the time series by the time column and also returns ``keys``
    that uniquely identify the time series. The last is useful for the
    final Spark Data Frame that will contain a forecast for each time
    series.

    """
    # Time series to be forecasted
    # The indentifiers for the time series
    keys = {
        grp_col: getattr(data.iloc[0], grp_col) for grp_col in group_columns
    }
    # We ensure datetime as pandas
    data[time_column] = pd.to_datetime(data[time_column])
    # We set time index
    data = data.set_index(time_column).sort_index()
    # We return prepared time series and its keys
    return data[target_column], keys


# Decorate the forecast functions to enable Spark API suitable returns
_forecast = return_dataframe_for_spark_api(forecast)
_multivariate_forecast = return_dataframe_for_spark_api(multivariate_forecast)
_probabilistic_forecast = return_dataframe_probabilistic_for_spark_api(
    probabilistic_forecast
)
_probabilistic_multivariate_forecast = (
    return_dataframe_probabilistic_for_spark_api(
        probabilistic_multivariate_forecast
    )
)


def _forecast_grouped_data(
    data: pd.DataFrame,
    time_column,
    target_column: str,
    group_columns: List[str],
    model_name: str,
    freq: str,
    steps: int,
    model_kwargs: Optional[Dict],
) -> pd.Series:
    """Applies local forecast in grouped data (This is a pandas UDF)

    Applies :func:`spark_forecast.models.local_forecasting.forecast`
    in data comming from a Spark Data Frame group by operation
    with an applyInPandas.


    Parameters
    ----------
    data
        Grouped data comming from groupby operation.
    time_column
        Name of the time column in the SparkDataFrame.
    target_column
        Name of the target column in the Spark DataFrame
        representing de time dependent variable in the time series.
    group_columns
        Name or a list of columns to perform group by operations
        identifying uniquely the time series in the Spark DataFrame.
    model_name
        Name of the model to select.
    freq
        A string with a valid pandas frequency (Example for day: 'D')
        of the time series in ``data``.
    steps
        Number of future predictions.
    model_kwargs
        Diccionary for initialization (passed to `__init__`) of the
        model given by ``model_name``.

    """

    # Time series to be forecasted and its keys

    y, keys = _prepare_grouped_data(
        data, target_column, time_column, group_columns
    )

    return _forecast(
        y=y,
        model_name=model_name,
        freq=freq,
        steps=steps,
        time_column=time_column,
        keys=keys,
        model_kwargs=model_kwargs,
    )


def _multivariate_forecast_grouped_data(
    data: pd.DataFrame,
    time_column,
    target_column: str,
    group_columns: List[str],
    model_name: str,
    freq: str,
    steps: int,
    future_covariates: Optional[Union[pd.DataFrame, pd.Series]],
    model_kwargs: Optional[Dict],
) -> pd.Series:
    """Applies multivariate forecast in grouped data (This is a pandas UDF)

    Applies :func:`spark_forecast.models.multivariate_forecasting.multivariate_forecast`
    in data comming from a Spark Data Frame group by operation.


    Parameters
    ----------
    data
        Grouped data comming from groupby operation.
    time_column
        Name of the time column in the SparkDataFrame.
    target_column
        Name of the target column in the Spark DataFrame
        representing de time dependent variable in the time series.
    group_columns
        Name or a list of columns to perform group by operations
        identifying uniquely the time series in the Spark DataFrame.
    model_name
        Name of the model to select.
    freq
        A string with a valid pandas frequency (Example for day: 'D')
        of the time series in ``data``.
    steps
        Number of future predictions.
    model_kwargs
        Diccionary for initialization (passed to `__init__`) of the
        model given by ``model_name``.

    """

    # Time series to be forecasted and its keys
    y, keys = _prepare_grouped_data(
        data, target_column, time_column, group_columns
    )

    return _multivariate_forecast(
        y=y,
        model_name=model_name,
        future_covariates=future_covariates,
        freq=freq,
        steps=steps,
        time_column=time_column,
        keys=keys,
        model_kwargs=model_kwargs,
    )


def _probabilistic_forecast_grouped_data(
    data: pd.DataFrame,
    time_column,
    target_column: str,
    group_columns: List[str],
    model_name: str,
    freq: str,
    steps: int,
    num_samples: int,
    lower_quantile: float,
    upper_quantile: float,
    model_kwargs: Optional[Dict],
):

    # Time series to be forecasted and its keys
    y, keys = _prepare_grouped_data(
        data, target_column, time_column, group_columns
    )
    return _probabilistic_forecast(
        y=y,
        model_name=model_name,
        freq=freq,
        steps=steps,
        num_samples=num_samples,
        keys=keys,
        time_column=time_column,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        model_kwargs=model_kwargs,
    )


def _probabilistic_multivariate_forecast_grouped_data(
    data: pd.DataFrame,
    time_column,
    target_column: str,
    group_columns: List[str],
    model_name: str,
    freq: str,
    steps: int,
    num_samples: int,
    lower_quantile: float,
    upper_quantile: float,
    future_covariates: Optional[pd.DataFrame],
    model_kwargs: Optional[Dict],
):

    # Time series to be forecasted and its keys
    y, keys = _prepare_grouped_data(
        data, target_column, time_column, group_columns
    )
    return _probabilistic_multivariate_forecast(
        y=y,
        model_name=model_name,
        future_covariates=future_covariates,
        freq=freq,
        steps=steps,
        num_samples=num_samples,
        keys=keys,
        time_column=time_column,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        model_kwargs=model_kwargs,
    )


def _forecast_grouped_data_with_classification(
    data: pd.DataFrame,
    time_column,
    target_column: str,
    group_columns: List[str],
    freq: str,
    steps: int,
    future_covariates: Optional[Union[pd.DataFrame, pd.Series]],
    model_classification_dict: Optional[Dict],
):

    # We get the model for this grouped data
    model_kwargs = model_classification_dict[
        data.iloc[0].classification
    ].copy()

    if model_kwargs["model_name"] in MODELS:
        return _forecast_grouped_data(
            data=data,
            time_column=time_column,
            target_column=target_column,
            group_columns=group_columns,
            model_name=model_kwargs.pop("model_name"),
            steps=steps,
            freq=freq,
            model_kwargs=model_kwargs,
        )
    return _multivariate_forecast_grouped_data(
        data=data,
        time_column=time_column,
        target_column=target_column,
        group_columns=group_columns,
        model_name=model_kwargs.pop("model_name"),
        steps=steps,
        freq=freq,
        future_covariates=future_covariates,
        model_kwargs=model_kwargs,
    )


def forecast_spark(
    data: pyspark.sql.dataframe.DataFrame,
    target_column: str,
    time_column: str,
    group_columns: Union[str, List[str]],
    model_name: str,
    freq: str,
    steps: int = 1,
    model_kwargs: Optional[Dict] = None,
) -> pyspark.sql.dataframe.DataFrame:
    """Forecast of multiple time series on Spark.

    Given a Spark DataFrame ``data`` containing
    multiples time series that can be identified
    by the unique keys ``group_columns``.


    Returns
    -------
        An Spark DataFrame containing a number of
        ``steps`` predictions for each time series
        that are uniquely identified by ``group_columns``.

    """
    model_kwargs = __fairy_default_freq(model_name, model_kwargs, freq)

    # We validate that columns passed exists in the DataFrame
    __validate_common_params(
        steps, time_column, target_column, group_columns, list(data.columns)
    )
    # Before going into spark, we check that the model
    # Can be initialized without problems
    __check_if_can_be_initialized(
        model_name,
        model_kwargs=model_kwargs,
    )
    # We perform forecast on the first time series found
    _ = forecast(
        __extract_first_serie(
            data,
            group_columns=group_columns,
            target_column=target_column,
            time_column=time_column,
        ),
        model_name=model_name,
        freq=freq,
        steps=steps,
        model_kwargs=model_kwargs,
    )

    # Finally we apply the forecast to each time series
    return data.groupby(group_columns).applyInPandas(
        lambda data: _forecast_grouped_data(
            data,
            time_column=time_column,
            target_column=target_column,
            group_columns=group_columns_to_list(group_columns),
            model_name=model_name,
            freq=freq,
            steps=steps,
            model_kwargs=model_kwargs,
        ),
        schema=__build_forecast_schema(time_column, group_columns),
    )


def multivariate_forecast_spark(
    data: pyspark.sql.dataframe.DataFrame,
    target_column: str,
    time_column: str,
    group_columns: Union[str, List[str]],
    model_name: str,
    freq: str,
    future_covariates: Optional[Union[pd.DataFrame, pd.Series]] = None,
    steps: int = 1,
    model_kwargs: Optional[Dict] = None,
) -> pyspark.sql.dataframe.DataFrame:
    """Forecast multivariate on Spark.

    Given a Spark DataFrame ``data`` containing
    multiples time series that can be identified
    by the unique keys ``group_columns``.


    Returns
    -------
        An Spark DataFrame containing a number of
        ``steps`` predictions for each time series
        that are uniquely identified by ``group_columns``.

    """
    model_kwargs = __fairy_default_freq(model_name, model_kwargs, freq)

    # We validate that columns passed exists in the DataFrame
    __validate_common_params(
        steps, time_column, target_column, group_columns, list(data.columns)
    )
    # Before going into spark, we check that the model
    # Can be initialized without problems
    __check_if_can_be_initialized(
        model_name,
        model_kwargs=model_kwargs,
    )

    # We perform forecast on the first time series found in order to raise possible errors
    _ = multivariate_forecast(
        __extract_first_serie(
            data,
            group_columns=group_columns,
            target_column=target_column,
            time_column=time_column,
        ),
        model_name=model_name,
        freq=freq,
        steps=steps,
        future_covariates=future_covariates,
        model_kwargs=model_kwargs,
    )
    # Finally we apply the forecast to each time series
    return data.groupby(group_columns).applyInPandas(
        lambda data: _multivariate_forecast_grouped_data(
            data,
            time_column=time_column,
            target_column=target_column,
            group_columns=group_columns_to_list(group_columns),
            model_name=model_name,
            future_covariates=future_covariates,
            freq=freq,
            steps=steps,
            model_kwargs=model_kwargs,
        ),
        schema=__build_forecast_schema(time_column, group_columns),
    )


def probabilistic_forecast_spark(
    data: pyspark.sql.dataframe.DataFrame,
    target_column: str,
    time_column: str,
    group_columns: Union[str, List[str]],
    model_name: str,
    freq: str,
    steps: int = 1,
    num_samples: int = 20,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
    model_kwargs: Optional[Dict] = None,
) -> pyspark.sql.dataframe.DataFrame:
    """Probabilistic Forecast of multiple time series on Spark.

    Given a Spark DataFrame ``data`` containing
    multiples time series that can be identified
    by the unique keys ``group_columns`` it performs
    a probabilistic forecast for each one of them,
    returning a probabilistic band of forecast.


    Returns
    -------
        An Spark DataFrame containing a number of
        ``steps`` predictions for each time series
        that are uniquely identified by ``group_columns``.

    """

    model_kwargs = __fairy_default_freq(model_name, model_kwargs, freq)

    __check_if_can_be_initialized(
        model_name,
        model_kwargs=model_kwargs,
        check_probabilistic=True,
    )

    __validate_common_params(
        steps,
        time_column,
        target_column,
        group_columns,
        list(data.columns),
        upper_quantile=upper_quantile,
        lower_quantile=lower_quantile,
    )
    # We perform forecast on the first time series found in order to raise possible errors
    _ = probabilistic_forecast(
        __extract_first_serie(
            data,
            group_columns=group_columns,
            target_column=target_column,
            time_column=time_column,
        ),
        model_name=model_name,
        freq=freq,
        steps=steps,
        num_samples=num_samples,
        model_kwargs=model_kwargs,
    )

    return data.groupby(group_columns).applyInPandas(
        lambda data: _probabilistic_forecast_grouped_data(
            data,
            model_name=model_name,
            time_column=time_column,
            target_column=target_column,
            group_columns=group_columns_to_list(group_columns),
            freq=freq,
            steps=steps,
            num_samples=num_samples,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            model_kwargs=model_kwargs,
        ),
        schema=__build_forecast_probabilistic_schema(
            time_column, group_columns, lower_quantile, upper_quantile
        ),
    )


def probabilistic_multivariate_forecast_spark(
    data: pyspark.sql.dataframe.DataFrame,
    target_column: str,
    time_column: str,
    group_columns: Union[str, List[str]],
    model_name: str,
    freq: str,
    future_covariates: Optional[Union[pd.DataFrame, pd.Series]] = None,
    steps: int = 1,
    num_samples: int = 20,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
    model_kwargs: Optional[Dict] = None,
) -> pyspark.sql.dataframe.DataFrame:
    """Probabilistic Forecast of multiple time series on Spark.

    Given a Spark DataFrame ``data`` containing
    multiples time series that can be identified
    by the unique keys ``group_columns`` it performs
    a probabilistic forecast for each one of them,
    returning a probabilistic band of forecast.


    Returns
    -------
        An Spark DataFrame containing a number of
        ``steps`` predictions for each time series
        that are uniquely identified by ``group_columns``.

    """

    model_kwargs = __fairy_default_freq(model_name, model_kwargs, freq)

    __check_if_can_be_initialized(
        model_name,
        model_kwargs=model_kwargs,
        check_probabilistic=True,
    )

    __validate_common_params(
        steps,
        time_column,
        target_column,
        group_columns,
        list(data.columns),
        upper_quantile=upper_quantile,
        lower_quantile=lower_quantile,
    )

    # We perform forecast on the first time series found in order to raise possible errors
    _ = probabilistic_multivariate_forecast(
        __extract_first_serie(
            data,
            group_columns=group_columns,
            target_column=target_column,
            time_column=time_column,
        ),
        model_name=model_name,
        freq=freq,
        steps=steps,
        future_covariates=future_covariates,
        model_kwargs=model_kwargs,
        num_samples=num_samples,
    )

    return data.groupby(group_columns).applyInPandas(
        lambda data: _probabilistic_multivariate_forecast_grouped_data(
            data,
            model_name=model_name,
            future_covariates=future_covariates,
            time_column=time_column,
            target_column=target_column,
            group_columns=group_columns_to_list(group_columns),
            freq=freq,
            steps=steps,
            num_samples=num_samples,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            model_kwargs=model_kwargs,
        ),
        schema=__build_forecast_probabilistic_schema(
            time_column, group_columns, lower_quantile, upper_quantile
        ),
    )


def forecast_with_classification_spark(
    data: pyspark.sql.dataframe.DataFrame,
    target_column: str,
    time_column: str,
    group_columns: Union[str, List[str]],
    freq: str,
    classification_data: pyspark.sql.dataframe.DataFrame,
    model_classification_dict: Dict[str, Dict],
    future_covariates: Optional[Union[pd.DataFrame, pd.Series]] = None,
    steps: int = 1,
) -> pyspark.sql.dataframe.DataFrame:
    """Forecast of multiple time series on Spark.

    Given a Spark DataFrame ``data`` containing
    multiples time series that can be identified
    by the unique keys ``group_columns``.


    Returns
    -------
        An Spark DataFrame containing a number of
        ``steps`` predictions for each time series
        that are uniquely identified by ``group_columns``.

    """

    # We validate that columns passed exists in the DataFrame
    __validate_common_params(
        steps, time_column, target_column, group_columns, list(data.columns)
    )
    # Before going into spark, we check that the model
    # Can be initialized without problems
    __validate_model_dict_with_classification(
        classification_data, model_classification_dict, freq=freq
    )

    data_with_classification = data.join(
        classification_data, on=group_columns, how="inner"
    )
    # We will validate now on the first series found
    __classification_of_first_serie = (
        data_with_classification.first().classification
    )
    # We extract the first serie
    __first_serie = __extract_first_serie(
        data,
        group_columns=group_columns,
        target_column=target_column,
        time_column=time_column,
    )
    __model_kwargs = model_classification_dict[
        __classification_of_first_serie
    ].copy()

    if __model_kwargs["model_name"] in MODELS:
        _ = forecast(
            __first_serie,
            model_name=__model_kwargs.pop("model_name"),
            freq=freq,
            steps=steps,
            model_kwargs=__model_kwargs,
        )
    else:
        _ = multivariate_forecast(
            __first_serie,
            model_name=__model_kwargs.pop("model_name"),
            freq=freq,
            steps=steps,
            future_covariates=future_covariates,
            model_kwargs=__model_kwargs,
        )

    # Finally we apply the forecast to each time series
    return data_with_classification.groupby(group_columns).applyInPandas(
        lambda data: _forecast_grouped_data_with_classification(
            data,
            time_column=time_column,
            target_column=target_column,
            group_columns=group_columns_to_list(group_columns),
            freq=freq,
            steps=steps,
            model_classification_dict=model_classification_dict,
            future_covariates=future_covariates,
        ),
        schema=__build_forecast_schema(time_column, group_columns),
    )
