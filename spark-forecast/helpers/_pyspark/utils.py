from functools import wraps
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
import pyspark
from pyspark.sql.types import DateType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import FloatType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType

from spark_forecast.exc import ColumnNamesError
from spark_forecast.exc import change_exception
from spark_forecast.helpers import logger


def group_columns_to_list(group_columns: Union[List[str], str]):
    return (
        group_columns if isinstance(group_columns, list) else [group_columns]
    )


def check_if_columns_in_dataframe(
    data: pyspark.sql.dataframe.DataFrame,
    columns: Union[List[str], str],
    extra_msg: Optional[str] = None,
):
    columns = group_columns_to_list(columns)
    if not set(columns) <= set(data.columns):
        if extra_msg:
            logger.error(extra_msg)
        raise ColumnNamesError(columns=data.columns, requested_columns=columns)


# TODO: Arreglar validador de schema


def pyspark_validate_schema(
    data: pyspark.sql.dataframe.DataFrame,
    target_column: str,
    time_column: str,
    *string_columns,
) -> None:
    """Validate schema for a simple dataset.

    A simple data set means that ``target_column`` is the column of
    interest to do an action (like cleaning or forecasting) and
    ``time_column`` is time index of the time series. Another columns
    passed on ``string_columns`` are assumed to be string.

    """

    error_msg = "Column `{}` hasn't valid data type, which must be {}."
    # Found schema: name:dtype in data
    found_schema = {field.name: field.dataType for field in data.schema.fields}

    with change_exception(
        ColumnNamesError(
            columns=data.columns,
            requested_columns=[target_column, time_column, *string_columns],
        ),
        KeyError,
    ):
        for name in string_columns:
            if not isinstance(found_schema[name], StringType):
                raise TypeError(error_msg.format(name, "String data type"))

        if not isinstance(found_schema[time_column], (StringType, DateType)):
            raise TypeError(
                error_msg.format(time_column, "String or Date data type")
            )
        if not isinstance(
            found_schema[target_column], (IntegerType, FloatType, DoubleType)
        ):
            raise TypeError(
                error_msg.format(
                    target_column, "Integer, Float or Double data type"
                )
            )


def return_dataframe_for_spark_api(forecast_function: Callable) -> Callable:
    @wraps(forecast_function)
    def wrapper_decorator(*args, **kwargs):
        try:

            time_column: str = kwargs.pop("time_column")
            keys: Dict[str, str] = kwargs.pop("keys")

            predictions = forecast_function(*args, **kwargs)
            return pd.DataFrame(
                {
                    "predictions": predictions,
                    time_column: predictions.index,
                    "error": None,
                    "type_error": None,
                    **keys,
                }
            )

        # Darts throws ValueError and TypeError. FairyForecastBaseException is custom
        # pylint: disable=broad-exception
        except Exception as exc:
            return pd.DataFrame(
                {
                    "predictions": [None],
                    time_column: [None],
                    "error": str(exc),
                    "type_error": str(type(exc)),
                    **keys,
                }
            )

    return wrapper_decorator


def return_dataframe_probabilistic_for_spark_api(
    forecast_function: Callable,
) -> Callable:
    @wraps(forecast_function)
    def wrapper_decorator(*args, **kwargs):
        try:

            time_column: str = kwargs.pop("time_column")
            keys: Dict[str, str] = kwargs.pop("keys")
            predictions = forecast_function(*args, **kwargs)
            return pd.DataFrame(
                {
                    **predictions.to_dict("list"),
                    time_column: predictions.index,
                    "error": None,
                    "type_error": None,
                    **keys,
                }
            )
        # Darts trough ValueError and TypeError. FairyForecastBaseException is custom
        # pylint: disable=broad-exceptio
        except Exception as exc:
            return pd.DataFrame(
                {
                    "predictions": [None],
                    f"lower_quantile_{kwargs['lower_quantile']}": [None],
                    f"upper_quantile_{kwargs['upper_quantile']}": [None],
                    time_column: [None],
                    "error": str(exc),
                    "type_error": str(type(exc)),
                    **keys,
                }
            )

    return wrapper_decorator


def validate_classification_dict(
    classification_data: pyspark.sql.dataframe.DataFrame,
    fill_dict: Dict[str, Dict],
):
    if "classification" not in classification_data.columns:
        raise ValueError(
            "Spark DataFrame must contain a column called `classification`"
            " with the series classification"
        )

    # Unique registers
    classifications = (
        classification_data.select("classification")
        .distinct()
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    # If any classification is not detailed in fill_dict, we raise
    if not set(classifications) <= set(fill_dict.keys()):
        raise ValueError(
            "There exists classification in the `classification_data` that hasn't found "
            f" on the `fill_dict` dictirionary whose keys are: {list(fill_dict.keys())}"
            f" and classifications found in `data` are: {classifications}"
        )
