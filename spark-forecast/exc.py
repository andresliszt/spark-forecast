# -*- coding: utf-8 -*-
"""Exceptions handling."""

import abc
from contextlib import contextmanager
from typing import Any
from typing import Generator
from typing import Type
from typing import Union


class FairyForecastBaseException(abc.ABC, BaseException):
    """Base class for custom errors and exceptions.
    Example:
        >>> class MyError(FairyForecastBaseException):
                msg_template = "Value ``{value}`` could not be found"
        >>> raise MyError(value="can't touch this")
        (...)
        MyError: Value `can't touch this` could not be found

    """

    @property
    @abc.abstractmethod
    def msg_template(self) -> str:
        """Un template para imprimir cuando una excepción es levantada.

        Example:
            "Value ``{value}`` is not found "

        """

    def __init__(self, **ctx: Any) -> None:
        self.ctx = ctx
        super().__init__()

    def __str__(self) -> str:
        txt = self.msg_template
        for name, value in self.ctx.items():
            txt = txt.replace("{" + name + "}", str(value))
        txt = txt.replace("`{", "").replace("}`", "")
        return txt


@contextmanager
def change_exception(
    raise_exc: Union[
        FairyForecastBaseException, Type[FairyForecastBaseException]
    ],
    *except_types: Type[BaseException],
) -> Generator[None, None, None]:
    """Context Manager para remplazar excepciones por propias.

    See also:
        :func:`pydantic.utils.change_exception`

    """
    try:
        yield
    except except_types as exception:
        raise raise_exc from exception  # type: ignore


class EnvVarNotFound(FairyForecastBaseException, NameError):
    """Raise this when a table name has not been found."""

    msg_template = "Enviroment variable `{env_var}` can't be found"


class MetricDifferentLengthError(FairyForecastBaseException, NameError):
    """Raise this when actual and predicted array have different length."""

    msg_template = (
        "Length actual `{len_actual}` and length predicted `{len_predicted}`"
    )


class NotDateTimeIndexError(FairyForecastBaseException, NameError):
    msg_template = (
        "Target serie 'y' must has DatetimeIndex. Got '{type_index}'"
    )


class InvalidDateTimeFrequency(FairyForecastBaseException, NameError):
    """Raise this when datetime index has different freq than desired (or freq None)"""

    msg_template = "Time index hasn't valid frequency for `{freq}`"


class DateTimeStringError(FairyForecastBaseException, NameError):

    msg_template = "String dates representation are malformed. Strings `{string_datetimes}`, freq `{freq}`"


class CovariatesTimeIndexDiffeError(FairyForecastBaseException, NameError):

    msg_template = "Covariates time index differs. Start_time: `{start_time}`, current_time: `{current_time}`, steps: `{}` and freq: `{freq}`"


class MinimumNumberOfObservationsError(FairyForecastBaseException, NameError):
    msg_template = "Time index with freq `{freq}` hasn't minimum number of observations to extract `{dt_feature}`. Required `{required}`"


class ColumnNamesError(FairyForecastBaseException, NameError):
    msg_template = "Data Frame has columns `{columns}` and requesting for `{requested_columns}`"


class TooShortTimeSeries(FairyForecastBaseException, NameError):
    msg_template = "Time Series is too short for method. length_received: `{length_received}` and length_minimum: `{length_minimum}`"


class TimeSeriesWithNullValues(FairyForecastBaseException, NameError):
    msg_template = "Time Series has null values."


class CovariatesWithNullValues(FairyForecastBaseException, NameError):
    msg_template = "Covariates has null values."


class TimeSeriesAndCovariatesIndexError(FairyForecastBaseException, NameError):
    """Raise this when covariates and time series have different indices."""

    msg_template = "Time Series and covariates have different indices."


class CustomPredictorsWithNullValues(FairyForecastBaseException, NameError):
    """Raise this when covariates and time series have different indices."""

    msg_template = "Custom predictors are returning nans."


class RegressorNotFittedError(FairyForecastBaseException, NameError):
    msg_template = "Trying to predict using a regressor that hasn't been fitted. Call `fit` method first"


class NotProbabilisticModelError(FairyForecastBaseException, NameError):
    msg_template = "Model `{type_model}` doesn't admit probabilistic forecast."


class ModelInitializationError(FairyForecastBaseException, NameError):
    """Raise this when a model can't be initialized."""

    msg_template = (
        "Model `{type_model}` can't be initialized with `{arguments}`"
    )
