# -*- coding: utf-8 -*-
"""Preprocessing utilities."""

import enum
from calendar import monthrange
from typing import List

import numpy as np
import pandas as pd

from spark_forecast.exc import InvalidDateTimeFrequency
from spark_forecast.exc import MinimumNumberOfObservationsError
from spark_forecast.exc import NotDateTimeIndexError
from spark_forecast.exc import TimeSeriesWithNullValues
from spark_forecast.exc import change_exception


class FairyForecastFrequencies(enum.Enum):

    DAY = "D"
    WEEK = "7D"
    HOUR = "H"
    HALF_HOUR = "0.5H"


def check_time_series_not_null(y: pd.Series):
    if y.isnull().any():
        raise TimeSeriesWithNullValues


def check_time_index(time_index, freq: str) -> None:
    """Check if time index has frequency `freq`

    Also, this method set the frequency `freq` as attribute on the
    index.

    """
    if not isinstance(time_index, pd.core.indexes.datetimes.DatetimeIndex):
        raise NotDateTimeIndexError(type_index=str(type(time_index)))

    with change_exception(InvalidDateTimeFrequency(freq=freq), ValueError):
        # This work as validation for time format
        time_index.freq = freq


def check_is_datetime_indexed_and_frequency(y: pd.Series, freq: str) -> None:
    """Check for suitable time series to be forecasted.

    Raises
    ------
    TypeError
       If ``y`` isn't pandas Series.
    TimeSeriesWithNullValues
        If ``y`` has null values.
    NotDateTimeIndexError
        If the index isn't :class:`pd.core.indexes.datetimes.DatetimeIndex`.
    InvalidDateTimeFrequency
        If the index frequency isn't ``freq``.

    """

    if not isinstance(y, pd.Series):
        raise TypeError(f"`y` must be a pandas Series. Got {type(y)}")

    check_time_series_not_null(y)

    check_time_index(y.index, freq)


# TODO: Pasar a un 'dict freq: Cantidad mínima de observaciones'


class DatesFeatures:
    """Dates features logic.

    This is a helper class to build dates features matrix.
    validations are performed to ensure that the number of observations
    in the time series matches the given frequency ``freq``
    so as not to cause errors when introducing predictors with
    insufficient temporal information. For example, if in a given
    series we introduce the months as predictor, if we only observed
    the first 3 months (January, February and March) and we want to
    predict in April, the model won't know what is April.

    For instance if we want to extract a feature, we need enough
    observation to be possible.

    """

    def __init__(self, freq: FairyForecastFrequencies) -> None:
        self.freq = freq

    def _validates_at_least_one_year(
        self, time_index: pd.core.indexes.datetimes.DatetimeIndex
    ) -> None:
        """Validates a year observations according to :py:attr:`~freq`."""

        if (
            self.freq == FairyForecastFrequencies.DAY.value
            and len(time_index) <= 365
        ):
            # If daily frequency we impose 366 days (it includes leap-year)
            raise MinimumNumberOfObservationsError(
                freq=self.freq, required=366, dt_feature="month"
            )
        if (
            self.freq == FairyForecastFrequencies.WEEK.value
            and len(time_index) <= 52
        ):
            # If weekly frequency, we impose 53 week (it includes leap-year)
            raise MinimumNumberOfObservationsError(
                freq=self.freq, required=53, dt_feature="month"
            )
        if (
            self.freq == FairyForecastFrequencies.HOUR.value
            and len(time_index) <= 8784
        ):
            # If hourly frequency then 366*24 = 8784
            raise MinimumNumberOfObservationsError(
                freq=self.freq, required=8785, dt_feature="month"
            )
        if (
            self.freq == FairyForecastFrequencies.HALF_HOUR.value
            and len(time_index) <= 17568
        ):
            # If frequency every 30 minutes then 366*24*2 = 17568
            raise MinimumNumberOfObservationsError(
                freq=self.freq, required=17569, dt_feature="month"
            )

    def _validates_at_least_one_month(self, time_index):
        """Validates a month observations according to :py:attr:`~freq`."""

        if (
            self.freq == FairyForecastFrequencies.DAY.value
            and len(time_index) <= 31
        ):
            raise MinimumNumberOfObservationsError(
                freq=self.freq, required=32, dt_feature="week"
            )

        if (
            self.freq == FairyForecastFrequencies.WEEK.value
            and len(time_index) <= 4
        ):
            raise MinimumNumberOfObservationsError(
                freq=self.freq, required=5, dt_feature="week"
            )

        if (
            self.freq == FairyForecastFrequencies.HOUR.value
            and len(time_index) <= 744
        ):
            raise MinimumNumberOfObservationsError(
                freq=self.freq, required=745, dt_feature="week"
            )

        if (
            self.freq == FairyForecastFrequencies.HALF_HOUR.value
            and len(time_index) <= 1484
        ):
            raise MinimumNumberOfObservationsError(
                freq=self.freq, required=1485, dt_feature="week"
            )

    @staticmethod
    def _number_of_week(day: int) -> int:
        """Given a ``day`` of the month (1-31) returns custom week classification.

        This class considers 3 types of parts of the month: The begining
        of the month (0), the middle (1) and the end of the month (2).

        """
        if day <= 7:
            # This means the first seven day of the month
            return 0
        if 7 < day <= 22:
            # This means the middle
            return 1
        # This means end of the month
        return 2

    def extract_month(
        self, time_index: pd.core.indexes.datetimes.DatetimeIndex
    ) -> pd.core.indexes.numeric.Int64Index:
        """Extracts month feature (1-12) from ``time_index``

        It will used in a Machine learning model using month feature as
        covariate. Note that if you want to use month feature, you must
        have enough observations otherwise the model could fail in its
        prediction, for example, suppose that you have time series
        containing date information from 1 January until 31 July, if the
        model is training using those months, the next month August is
        unknown for the model and prediction will fail. Therefore,
        multiples validations are made to prevent this unexpected
        behaivour and are related to the frequency :py:attr:`~freq`.

        """
        self._validates_at_least_one_year(time_index)

        return time_index.month

    def extract_week(
        self, time_index: pd.core.indexes.datetimes.DatetimeIndex
    ) -> pd.core.indexes.numeric.Int64Index:
        """Extracts custom week feature (0-2) from ``time_index``

        It will used in a Machine learning model using
        custom week feature as covariate. Note that if you want
        to use month feature, you must have enough observations
        otherwise the model could fail in its prediction.

        See :py:meth:`~_number_of_week` method.

        """
        self._validates_at_least_one_month(time_index)

        return time_index.day.map(self._number_of_week)

    def extract_dayofweek(
        self, time_index: pd.core.indexes.datetimes.DatetimeIndex
    ) -> pd.core.indexes.numeric.Int64Index:
        """Extracts day of week feature (0-6) from ``time_index``

        It will used in a Machine learning model using
        month feature as covariate. Note that if you want
        to use day feature, you must have enough observations
        otherwise the model could fail in its prediction.

        This method indentifies Monday with 0 and so on
        till Sunday with 6.

        For weekly frequencies this method doesn't allow
        getting the day, because weekly frequencies usally
        contains only one day (for example, weekly frequencies
        of Mondays).

        """

        if (
            self.freq == FairyForecastFrequencies.DAY.value
            and len(time_index) <= 6
        ):
            raise MinimumNumberOfObservationsError(
                freq=self.freq, required=7, dt_feature="day"
            )

        if self.freq == FairyForecastFrequencies.WEEK.value:
            raise ValueError(
                "If series is indexed by week, day feature is not allowed"
            )

        if (
            self.freq == FairyForecastFrequencies.HOUR.value
            and len(time_index) <= 168
        ):
            raise MinimumNumberOfObservationsError(
                freq=self.freq, required=169, dt_feature="day"
            )

        if (
            self.freq == FairyForecastFrequencies.HALF_HOUR.value
            and len(time_index) <= 336
        ):
            raise MinimumNumberOfObservationsError(
                freq=self.freq, required=337, dt_feature="day"
            )

        return time_index.dayofweek

    def extract_dayofmonth(
        self, time_index: pd.core.indexes.datetimes.DatetimeIndex
    ) -> pd.core.indexes.numeric.Int64Index:
        """Extracts day of week feature (1-31) from ``time_index``."""
        self._validates_at_least_one_month(time_index)

        return time_index.day

    def extract_weekofyear(
        self, time_index: pd.core.indexes.datetimes.DatetimeIndex
    ) -> pd.core.indexes.numeric.Int64Index:
        """Returns week of year (1-53).

        If 53 is reached is for leap-year.

        """
        self._validates_at_least_one_year(time_index)
        # weekofyear is deprecated
        return time_index.isocalendar()["week"].values

    def create_dates_features(
        self,
        time_index: pd.core.indexes.datetimes.DatetimeIndex,
        extract_columns: List[str],
    ) -> pd.DataFrame:
        """Create dates features for model input.

        Given ``time_index`` this method construct a dates
        features matrix for a given frequency ``freq``.
        ``time_index`` must be correct indexed for the frequency
        ``freq`` before use this method, otherwise exception will
        be raised.

        Examples
        --------
        >>> create_dates_features(
            pd.date_range("2018-01-01", periods=10),
            freq="D",
            extract_columns=["month", "week", "day"],
        )
                    month  weekofmonth  dayofweek
        2018-01-01      1            1          0
        2018-01-02      1            1          1
        2018-01-03      1            1          2
        2018-01-04      1            1          3
        2018-01-05      1            1          4
        2018-01-06      1            1          5
        2018-01-07      1            1          6
        2018-01-08      1            2          0
        2018-01-09      1            2          1
        2018-01-10      1            2          2


        >>> create_dates_features(pd.Index([pd.Timestamp('2012-05-10'), pd.Timestamp('2012-05-11')]))
                    month  weekofmonth  dayofweek
        2012-05-10      5            2          3
        2012-05-11      5            2          4

        >>> create_dates_features(pd.Index([pd.Timestamp('2012-05-10'), pd.Timestamp('2012-05-12')])))
        InvalidDateTimeFrequency: Time index hasn't valid frequency for `D`

        See also
        --------
        ``https://pandas.pydata.org/docs/user_guide/timeseries.html``


        Returns
        -------
        Dates features with columns ``month``, ``day``, ``dayofweek``.
        ``dayofweek`` has enumaration from 0 to 6, where 0 is for Monday
        and 6 is for Sunday. If ``freq=H`` also ``hour`` column will be
        included. If ``freq=0.5H`` also ``hour`` and ``minute`` columns
        will be included.

        Raises
        ------
        ValueError
            if ``freq`` is not ``D``, ``7D``, ``H`` or ``0.5H``

        InvalidDateTimeFrequency
            if ``freq`` can't be imposed to ``time_index``

        """

        if not set(extract_columns) <= {
            "month",
            "week",
            "dayofweek",
            "weekofyear",
            "dayofmonth",
            "hour",
            "minutes",
        }:
            raise ValueError(
                "Date features columns allowed are 'month', 'week',"
                "'dayofweek', 'weekofyear', 'dayofmonth', 'hour' and 'minute'"
            )

        with change_exception(
            InvalidDateTimeFrequency(freq=self.freq), ValueError
        ):
            # This work as validation for time format
            time_index.freq = self.freq

        if self.freq != "H" and "hour" in extract_columns:
            raise ValueError(
                "'hour' can't be extracted from index with no hourly frequency"
            )

        if self.freq != "0.5H" and "minute" in extract_columns:
            raise ValueError(
                "'minute' can't be extracted from index with no 0.5-hourly frequency"
            )

        dates_dict = {}

        if "month" in extract_columns:
            dates_dict["month"] = self.extract_month(time_index)

        if "week" in extract_columns:
            dates_dict["week"] = self.extract_week(time_index)

        if "weekofyear" in extract_columns:
            dates_dict["weekofyear"] = self.extract_weekofyear(time_index)

        if "dayofweek" in extract_columns:
            dates_dict["dayofweek"] = self.extract_dayofweek(time_index)

        if "dayofmonth" in extract_columns:
            dates_dict["dayofmonth"] = self.extract_dayofmonth(time_index)

        if "hour" in extract_columns:
            dates_dict["hour"] = time_index.hour

        if "minutes" in extract_columns:
            dates_dict["minutes"] = time_index.minute

        return pd.DataFrame(dates_dict, index=time_index).astype(int)

    @staticmethod
    def _polar_transformation(
        date_feature: pd.core.indexes.numeric.Int64Index, frac
    ):
        theta = 2 * np.pi * date_feature / frac
        return np.sin(theta), np.cos(theta)

    def create_dates_features_polar(
        self,
        time_index: pd.core.indexes.datetimes.DatetimeIndex,
        extract_columns: List[str],
    ) -> pd.DataFrame:
        """This function maps dates features to polar coordinates for model input.

        The advantage of this approach is that polar coordinates
        mantain the sequential nature of time data, which is
        useful for machine learning models.

        Given ``time_index`` this method construct a dates
        features matrix for a given frequency ``freq``.
        ``time_index`` must be correctly indexed for the frequency
        ``freq`` before using this method, otherwise an exception will
        be raised.

        Examples
        --------
        >>> create_dates_features_polar(pd.date_range("2018-01-01", periods = 10), freq = "D", extract_columns = ["month", "week", "day"])

        """

        dates_features = self.create_dates_features(
            time_index, extract_columns
        )

        dates_dict = {}

        # Month of year, 1 to 12
        if "month" in extract_columns:
            (
                dates_dict["month_x"],
                dates_dict["month_y"],
            ) = self._polar_transformation(dates_features["month"], frac=12)
        # Week, 0 to 2
        if "week" in extract_columns:
            (
                dates_dict["week_x"],
                dates_dict["week_y"],
            ) = self._polar_transformation(dates_features["week"], frac=3)

        # Week of year, 1 to 52
        if "weekofyear" in extract_columns:
            (
                dates_dict["weekofyear_x"],
                dates_dict["weekofyear_y"],
            ) = self._polar_transformation(
                dates_features["weekofyear"], frac=52
            )

        # Day of week, 0 o 6
        if "dayofweek" in extract_columns:
            (
                dates_dict["dayofweek_x"],
                dates_dict["dayofweek_y"],
            ) = self._polar_transformation(dates_features["dayofweek"], frac=7)

        # Day of month, range obtained from monthrange
        if "dayofmonth" in extract_columns:
            dayofmonth = self.extract_dayofmonth(time_index)
            num_days_per_month = [
                monthrange(Y, M)[1]
                for Y, M in zip(time_index.year, time_index.month)
            ]
            dates_dict["dayofmonth_x"] = np.sin(
                2 * np.pi * dayofmonth / num_days_per_month
            )
            dates_dict["dayofmonth_y"] = np.cos(
                2 * np.pi * dayofmonth / num_days_per_month
            )

        # Hour of the day
        if "hour" in extract_columns:
            timeofday = time_index.hour + time_index.minute / 24
            dates_dict["hour_x"] = np.sin(2 * np.pi * timeofday / 24)
            dates_dict["hour_y"] = np.cos(2 * np.pi * timeofday / 24)

        return pd.DataFrame(dates_dict, index=time_index).astype(float)
