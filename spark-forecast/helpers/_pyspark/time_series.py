# -*- coding: utf-8 -*-
"""Helpers functions for Spark Pandas API."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import pyspark

from spark_forecast.classification.demand_classifier import (
    NaiveDemandClassifier,
)
from spark_forecast.exc import TooShortTimeSeries
from spark_forecast.exc import change_exception
from spark_forecast.helpers._pyspark.utils import group_columns_to_list
from spark_forecast.helpers._pyspark.utils import pyspark_validate_schema
from spark_forecast.helpers._pyspark.utils import validate_classification_dict
from spark_forecast.outlier_detection.outlier_detection import OUTLIERS_METHODS

# from spark_forecast.helpers import logger
from spark_forecast.preprocessing.utils import (
    check_is_datetime_indexed_and_frequency,
)

# pylint: disable=unnecessary-lambda


# TODO: Better outlier filler
# TODO: Dtype validators
# TODO: Filler based on conditions (for example time series classification)


class SparkTimeSeriesHelper:
    """Helpers for groupBy operations on Spark.

    All methods of this class are for apply operations on groupBy on
    Spark. A list of columns given by ``group_columns`` uniquely
    identify the time series where the atomic make sense (for example
    fill null values).

    """

    def __init__(
        self,
        freq: str,
        target_column: str,
        time_column: str,
        group_columns: Union[List[str], str],
    ) -> None:
        """Acts on a :class:`pyspark.sql.dataframe.DataFrame`
        containing information of multiples time series
        that can be uniquely identified by ``group_columns``.

        Parameters
        ----------
        target_column
            Name of the target column in the Spark DataFrame
            representing de time dependent variable in the time series.
        time_column
            Name of the time column in the SparkDataFrame.
        group_columns
            Name or a list of columns to perform group by operations
            identifying uniquely the time series in the Spark DataFrame.

        """

        self.freq = freq
        self.target_column = target_column
        self.time_column = time_column
        self.group_columns = group_columns_to_list(group_columns)
        self.__schema = self.__build_schema()
        self.naive_classifier: NaiveDemandClassifier = NaiveDemandClassifier()

    def __build_schema(self) -> str:
        """Schema for returning Data Frames in Spark.

        For instance ``self.target_column`` is assumed to be float.

        """

        return (
            f"{self.target_column} double, {self.time_column} date,"
            + ",".join([col + " string" for col in self.group_columns])
        )

    @staticmethod
    def __mode_duplicate_strategy(s: pd.Series) -> float:
        """Used internally for mode in possible null series."""
        try:
            return s.mode().loc[0]
        except KeyError:
            return np.nan

    @staticmethod
    def repeat_bfill(y: pd.Series, date_to: str, freq: str):

        extended_index = pd.date_range(date_to, y.index[0], freq=freq)
        if extended_index.empty:
            return y

        return y.reindex(
            extended_index.append(y.index[1:]),
            fill_value=y.iloc[0],
        )

    @staticmethod
    def zero_bfill(y: pd.Series, date_to: str, freq: str):

        extended_index = pd.date_range(date_to, y.index[0], freq=freq)
        if extended_index.empty:
            return y

        return y.reindex(
            extended_index.append(y.index[1:]),
            fill_value=0,
        )

    @staticmethod
    def repeat_ffill(y: pd.Series, date_to: str, freq: str):

        extended_index = pd.date_range(y.index[-1], date_to, freq=freq)
        if extended_index.empty:
            return y

        return y.reindex(
            y.index[:-1].append(extended_index),
            fill_value=y.iloc[-1],
        )

    @staticmethod
    def fill_and_fix_time_series(
        y: pd.Series,
        freq: str,
        duplicate_strategy: str,
        fillna_strategy: str,
        interpolator: Optional[Dict[str, Any]] = None,
        repeat_ffill_to: Optional[str] = None,
        zero_bfill_to: Optional[str] = None,
        repeat_bfill_to: Optional[str] = None,
    ) -> pd.Series:
        """Completes missing dates in a given pandas Series.

        Given a time series with a frequency ``freq``, this method
        will resample missing dates. First, if a date is repeated
        if a date is repeated, it will join the records of that
        date into one with a criterion given by ``duplicate_strategy``.
        If dictionary ``interpolator`` is given, then this method will use
        :meth:`pandas.Series.interpolate` with using it as keyword args.

        Finally, this method fill nans with a criterion given by
        ``fillna_strategy`` (Note careful that fill nan is mandatory,
        even if interpolation is applied, because interpolate doesn't
        guarantee that all nans will be removed).

        This method has to be used careful, because if the given
        frequency ``freq`` isn't the serie's frequency, then
        :meth:`pandas.Series.resample` will try to build the new
        frequency using merge aggregations, that could result
        in unexpected results.

        The objective of this method is to build series with
        no missed dates (with respect to the given ``freq``)
        and with no nans values.


        Parameters
        ----------
        y
            Time series to be fixed.
        freq
            Time frequency of ``y``.
        duplicate_strategy
            Merge duplicated dates into one. Valid
            values are ``mean``, ``mode``, ``median`` or ``sum``.
        fillnan_strategy
            Pandas fillnan strategy. Valid
            values are ``mean``, ``median`` or ``zero``.

        interpolator
            Dictionary that will work as keyword arguments
            for :meth:`pandas.Series.interpolate`.


        Example
        -------
        >>> y = pd.Series(
                [1, 1, 1, 1, 1, 1],
                index=pd.to_datetime(
                    [
                        "2022-02-10",
                        "2022-02-11",
                        "2022-02-14",
                        "2022-02-14",
                        "2022-02-24",
                        "2022-02-16",
                    ]
                ),
            )
        >>> fill_and_fix_time_series(
                y,
                freq="D",
                duplicate_strategy="mode",
                fillna_strategy="zero",
            )
        2022-02-10    1.0
        2022-02-11    1.0
        2022-02-12    0.0
        2022-02-13    0.0
        2022-02-14    1.0
        2022-02-15    0.0
        2022-02-16    1.0
        2022-02-17    0.0
        2022-02-18    0.0
        2022-02-19    0.0
        2022-02-20    0.0
        2022-02-21    0.0
        2022-02-22    0.0
        2022-02-23    0.0
        2022-02-24    1.0
        Freq: D, dtype: float64


        Returns
        -------
            The time series will no missing dates and no nans.

        """

        if not isinstance(y, pd.Series):
            raise TypeError(
                "This method is only supported with `y` as a `pandas.Series`"
            )

        if duplicate_strategy == "sum":
            y = y.resample(freq).agg(lambda s: np.nan if s.empty else s.sum())

        elif duplicate_strategy == "mode":
            y = y.resample(freq).agg(
                SparkTimeSeriesHelper.__mode_duplicate_strategy
            )

        elif duplicate_strategy == "median":
            y = y.resample(freq).median()

        elif duplicate_strategy == "mean":
            y = y.resample(freq).mean()

        else:
            raise ValueError(
                "Supported `duplicate_strategy` are `mean`, `mode`, `median` or `sum`."
            )

        if interpolator is not None:
            try:
                # Not all time series can be interpolated
                # with any method. In order to protect
                # we do linear interpolation if fails
                y = y.interpolate(**interpolator)
            except ValueError:
                y = y.interpolate("linear")

        if fillna_strategy == "zero":
            y = y.fillna(0)

        elif fillna_strategy == "median":
            y = y.fillna(y[y.notnull()].median())

        elif fillna_strategy == "mean":
            y = y.fillna(y[y.notnull()].mean())

        else:
            raise ValueError(
                "Supported `fillna_strategy` are `mean`, `zero`, or `median`."
            )

        if repeat_ffill_to is not None:
            y = SparkTimeSeriesHelper.repeat_ffill(y, repeat_ffill_to, freq)

        if zero_bfill_to is not None:
            y = SparkTimeSeriesHelper.zero_bfill(y, zero_bfill_to, freq)

        elif repeat_bfill_to is not None:
            y = SparkTimeSeriesHelper.repeat_bfill(y, repeat_bfill_to, freq)

        return y

    def __first_row_info(self, first_row: pd.Series) -> Dict[str, str]:
        """Get keys of grouped Data Frame.

        Internal method to get keys on applyInPandas group by
        operations.

        """
        # We transform to string to ensure the self.__schema defined
        return {
            grp_col: str(getattr(first_row, grp_col))
            for grp_col in self.group_columns
        }

    def __index_by_time(self, data: pd.DataFrame) -> pd.DataFrame:
        """Index ``data`` by time.

        Internal method to ensure that the time series passed to
        :py:meth:`~fill_and_fix_time_series` has time index.

        """
        data[self.time_column] = pd.to_datetime(data[self.time_column])
        return data.set_index(self.time_column)

    def fill_and_fix_time_series_from_grouped_data(
        self,
        data: pd.DataFrame,
        duplicate_strategy: str,
        fillna_strategy: str,
        interpolator: Optional[Dict[str, Any]] = None,
        outliers_detection: Optional[Dict[str, Any]] = None,
        repeat_ffill_to: Optional[str] = None,
        zero_bfill_to: Optional[str] = None,
        repeat_bfill_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """Applies :py:meth:`~fill_and_fix_time_series` on grouped Data Frame.

        The grouped Data Frame is one of the formed during group by operations
        in pyspark given by :py:attr:`~group_columns`. And we returns a pandas
        Data Frame because is what pyspark DataFrame applyInPandas methods
        requires.

        Returns ``data[[self.target_column, self.time_column, *self.group_columns]]``
        with the series ``data[self.target_column]`` *fixed* by
        :py:meth:`~fill_and_fix_time_series` after passing by
        an outlier removal (if ``outliers_detection`` is not None),
        that for now, are only univariate methods.

        """
        # Get the keys of the time series
        groups_values = self.__first_row_info(data.iloc[0])
        # We index by time
        data = self.__index_by_time(data)
        # This series containing not-null values will be used in outlier detection
        y_notnull = data[self.target_column][
            data[self.target_column].notnull()
        ]

        if y_notnull.empty:
            raise ValueError(
                "We don't allow fully null time series, remove from Spark"
                f" DataFrame before apply this method. Group_values: {groups_values}"
            )

        # If outerliers detection dict we replace outliers according to the selected method
        if outliers_detection is not None:
            # We don't validate valid model name
            # here, only did in the Spark version
            _outliers_detection = outliers_detection.copy()
            method = OUTLIERS_METHODS[_outliers_detection.pop("method")]
            # The time series to remove outliers
            # Outlier methods needs series without nans
            # outliers is a pandas Series with 1 (if not outlier) and -1
            # (if outlier) with the same index as y
            outliers = method(y_notnull, **_outliers_detection)
            # For instance ouliers are only detected, so we fill as nan
            data.loc[
                outliers[outliers == -1].index, self.target_column
            ] = np.nan

        # At this point data[self.target_column] is a situable time series
        y_fixed = self.fill_and_fix_time_series(
            data[self.target_column],
            freq=self.freq,
            fillna_strategy=fillna_strategy,
            duplicate_strategy=duplicate_strategy,
            interpolator=interpolator,
            zero_bfill_to=zero_bfill_to,
            repeat_bfill_to=repeat_bfill_to,
            repeat_ffill_to=repeat_ffill_to,
        )
        # We return dataframe with situable cols and values
        # Cols and dtypes match with self.__schema
        return pd.DataFrame(
            {
                self.target_column: y_fixed.values,
                self.time_column: y_fixed.index,
                **groups_values,
            }
        )

    @staticmethod
    def build_weekly_aggregation(
        y: pd.Series,
        day: str,
    ) -> pd.Series:
        """Generate weekly aggregation series from series with daily frequency.

        This is a helper method to build a weekly time series
        from daily one. Only valid for situable daily indexed time
        series. Weekly resample is applied, where weeks starts
        at Monday. All register for a week are added.

        Parameters
        ----------
        y
            Time series with daily frequency without null values.

        Returns
        -------
            New time series with weekly frequency.

        """

        # This method only support suitables time series
        # With daily frequency
        check_is_datetime_indexed_and_frequency(y, freq="D")

        if len(y) < 7:
            raise TooShortTimeSeries(length_received=len(y), length_minimum=7)

        return y.resample(f"W-{day}", label="left", closed="left").sum()

    def build_weekly_aggregation_from_grouped_data(
        self,
        data: pd.DataFrame,
        day: str,
    ) -> pd.DataFrame:
        """Applies :meth:`self._build_weekly_aggregation` on grouped Data Frame.

        Used internally on groupBy operation on Spark.

        """
        groups_values = self.__first_row_info(data.iloc[0])
        data = self.__index_by_time(data)
        # Weekly series for target column
        y_weekly = self.build_weekly_aggregation(
            data[self.target_column], day=day
        )

        return pd.DataFrame(
            {
                self.target_column: y_weekly.values,
                self.time_column: y_weekly.index,
                **groups_values,
            }
        )

    def naive_time_series_classifier(self, data: pd.DataFrame) -> pd.DataFrame:
        """Naive classification of time series on grouped Data Frame.

        The grouped Data Frame is one of the formed during group by operations
        in pyspark given by :py:attr:`~group_columns`. And we returns a pandas
        Data Frame because is what pyspark DataFrame applyInPandas methods
        requires.

        It returns a DataFrame with a single row containing the
        classification of the time series


        The classifier can be found in
        :class:`spark_forecast.classification.demand_classifier.NaiveDemandClassifier`

        """

        groups_values = self.__first_row_info(data.iloc[0])
        data = self.__index_by_time(data)
        classification = self.naive_classifier.classify(
            data[self.target_column]
        )
        return pd.DataFrame(
            {
                "classification": [classification],
                **groups_values,
            }
        )

    def fill_and_fix_time_series_from_grouped_data_on_classification(
        self,
        data: pd.DataFrame,
        fill_dict: Dict[str, Dict],
        repeat_ffill_to: Optional[str] = None,
        zero_bfill_to: Optional[str] = None,
        repeat_bfill_to: Optional[str] = None,
    ):
        fill_kwargs = fill_dict[data.iloc[0].classification]
        return self.fill_and_fix_time_series_from_grouped_data(
            data,
            repeat_ffill_to=repeat_ffill_to,
            zero_bfill_to=zero_bfill_to,
            repeat_bfill_to=repeat_bfill_to,
            **fill_kwargs,
        )

    @staticmethod
    def __validate_datestring(date):
        with change_exception(
            ValueError(
                f"`date` {date} isn't a valid date string representation"
            ),
            ValueError,
        ):
            # Simple validation from pandas
            pd.date_range(date, periods=1)

    @staticmethod
    def __validate_fill_arguments(
        fillna_strategy: str,
        duplicate_strategy: str,
        outliers_detection: Optional[Dict] = None,
        interpolator: Optional[Dict] = None,
        repeat_ffill_to: Optional[str] = None,
        zero_bfill_to: Optional[str] = None,
        repeat_bfill_to: Optional[str] = None,
    ) -> None:
        """Validate arguments for fillers.

        Private member that validates all the argumets
        passed to :py:meth:`~fill_and_fix_time_series_from_grouped_data`

        """

        if fillna_strategy not in ("mean", "zero", "median"):
            raise ValueError(
                "Supported `fillna_strategy` are `mean`, `zero`, or `median`."
            )

        if duplicate_strategy not in ("mean", "mode", "sum", "median"):
            raise ValueError(
                "Supported `duplicate_strategy` are `mean`, `mode`, `median` or `sum`."
            )

        if outliers_detection is not None:
            if (
                not isinstance(outliers_detection, dict)
                or "method" not in outliers_detection
            ):
                raise ValueError(
                    "`outliers_detection` must be a dictionary with a `method` key"
                    f"with one of {list(OUTLIERS_METHODS.keys())} and the other keys"
                    "are passed to the method, that can be found on"
                    "spark_forecast.outliers_detection.outliers_detections module."
                )

            if outliers_detection["method"] not in OUTLIERS_METHODS:
                raise ValueError(
                    f"Must supply `method` as key on `outliers_detection`."
                    f"Options are {list(OUTLIERS_METHODS.keys())}"
                )

        if interpolator is not None:
            if not isinstance(interpolator, dict):
                raise ValueError(
                    "`interpolator` must be a dictionary with valid keyword arguments"
                    " passed to `pandas.DataFrame.interpolate`."
                )

            for invalid_kw in ("axis", "inplace"):
                # Not allowed kwargs
                if invalid_kw in interpolator:
                    del interpolator[invalid_kw]

        if sum([zero_bfill_to is not None, repeat_bfill_to is not None]) == 2:
            raise ValueError(
                "`zero_bfill_to` and `repeat_bfill_to` are mutually excluyent."
            )

        if zero_bfill_to is not None:
            SparkTimeSeriesHelper.__validate_datestring(zero_bfill_to)

        if repeat_bfill_to is not None:
            SparkTimeSeriesHelper.__validate_datestring(repeat_bfill_to)

        if repeat_ffill_to is not None:
            SparkTimeSeriesHelper.__validate_datestring(repeat_ffill_to)

    def __validate_fill_dict(
        self,
        classification_data: pyspark.sql.dataframe.DataFrame,
        fill_dict: Dict[str, Dict],
    ) -> None:
        """Validates synchronization in ``classification_data`` and ``fill_dict``.

        Private method that ensure that all unique classification found
        in ``classification`` are specified in ``fill_dict``.

        """

        validate_classification_dict(classification_data, fill_dict)
        # Validates that all inner dictionary found in fill_data
        # is a valid kwarg-dict for the method
        # fill_and_fix_time_series_from_grouped_data
        with change_exception(
            ValueError(
                "Wrong option passed to `fill_dict` inner dictionary."
                " Valid options are: `fillna_strategy`, `duplicate_strategy`,"
                " `interpolator` (Optional), `outliers_detection` (Optional)"
            ),
            TypeError,
        ):
            for fill_option in fill_dict.values():
                self.__validate_fill_arguments(**fill_option)

    def naive_time_series_classifier_spark(
        self, data: pyspark.sql.dataframe.DataFrame
    ) -> pyspark.sql.dataframe.DataFrame:
        """Naive Time Series classification on Spark.

        Applies :meth:`spark_forecast.classification.demand_classifier.classify`
        for each unique series given by :py:attr:`~group_columns`.


        Parameters
        ----------
        data
            Data as :class:`pyspark.sql.dataframe.DataFrame`

        Returns
        -------
            Returns ``data[["classification", *self.group_columns]]``,
            i.e the unique identifier per series plus an additional
            col with its classification.

        """

        pyspark_validate_schema(
            data, self.target_column, self.time_column, *self.group_columns
        )

        return data.groupBy(self.group_columns).applyInPandas(
            lambda d: self.naive_time_series_classifier(d),
            schema=",".join(
                [
                    col + " string"
                    for col in ["classification", *self.group_columns]
                ]
            ),
        )

    def build_weekly_aggregation_spark(
        self, data: pyspark.sql.dataframe.DataFrame, day: str = "MON"
    ) -> pyspark.sql.dataframe.DataFrame:
        """Transform a whole data as pyspark Dataframe into weekly aggregation.

        At this point, ``data`` must has perfect daily frequency for each
        unique series given by ``self.group_columns``. This is a helper
        function for time series that needs conversion to
        weekly frequency. Also, no time series must contain nulls,
        otherwise this method will fail. We recomend to use it
        after using :py:meth:`~fill_and_fix_time_series_spark` for
        fix daily frequency time series.

        Returns
        -------
            :class:`pyspark.pandas.DataFrame` with
            weekly time series given by ``group_columns``.

        """

        if day not in ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"):
            raise ValueError(
                "`day` must be one of `MON`, `TUE`, `WED`,`THU`, `FRI`, `SAT`, `SUN` "
            )

        pyspark_validate_schema(
            data, self.target_column, self.time_column, *self.group_columns
        )
        return data.groupBy(self.group_columns).applyInPandas(
            lambda d: self.build_weekly_aggregation_from_grouped_data(
                d, day=day
            ),
            schema=self.__schema,
        )

    def fill_and_fix_time_series_spark(
        self,
        data: pyspark.sql.dataframe.DataFrame,
        duplicate_strategy: str = "mode",
        fillna_strategy: str = "mean",
        interpolator: Optional[Dict[str, Any]] = None,
        outliers_detection: Optional[Dict[str, Any]] = None,
        repeat_ffill_to: Optional[str] = None,
        zero_bfill_to: Optional[str] = None,
        repeat_bfill_to: Optional[str] = None,
    ) -> pyspark.sql.dataframe.DataFrame:
        """Completes missing values for a whole data as pyspark Dataframe.

        Given a pyspark Dataframe, each time series to be filled
        is uniquely identified for ``group_columns``. ``time_column`` is
        a column for dates and ``target_column`` is the value that
        we want to forecast. This method (for the moment) doesn't pay
        attention on the other columns and will be ignored in the output,
        this means that will fill only for ``self.target_column`` and
        we will extend ``self.time_column`` if has missing dates.

        All the logic is defined on :py:meth:`~fill_and_fix_time_series`.
        Note careful that this method will return
        ``data[[self.target_column, self.time_column, *self.group_columns]]`` clean
        and ready to be forecasted.


        Parameters
        ----------
        data
            Data as :class:`pyspark.sql.dataframe.DataFrame`

        kwargs
        ------
            passed to :py:meth:`~fill_and_fix_time_series`

        Returns
        -------
            :class:`pyspark.pandas.DataFrame` with all
            time series given by ``group_columns`` filled.

        """

        pyspark_validate_schema(
            data, self.target_column, self.time_column, *self.group_columns
        )

        self.__validate_fill_arguments(
            fillna_strategy=fillna_strategy,
            duplicate_strategy=duplicate_strategy,
            outliers_detection=outliers_detection,
            interpolator=interpolator,
            repeat_ffill_to=repeat_ffill_to,
            zero_bfill_to=zero_bfill_to,
            repeat_bfill_to=repeat_bfill_to,
        )
        # pylint: disable=unnecessary-lambda
        return data.groupBy(self.group_columns).applyInPandas(
            lambda d: self.fill_and_fix_time_series_from_grouped_data(
                d,
                fillna_strategy=fillna_strategy,
                duplicate_strategy=duplicate_strategy,
                interpolator=interpolator,
                outliers_detection=outliers_detection,
                repeat_ffill_to=repeat_ffill_to,
                zero_bfill_to=zero_bfill_to,
                repeat_bfill_to=repeat_bfill_to,
            ),
            schema=self.__schema,
        )

    def fill_and_fix_time_series_spark_based_on_classification(
        self,
        data: pyspark.sql.dataframe.DataFrame,
        classification_data: pyspark.sql.dataframe.DataFrame,
        fill_dict: Dict[str, Dict],
        repeat_ffill_to: Optional[str] = None,
        zero_bfill_to: Optional[str] = None,
        repeat_bfill_to: Optional[str] = None,
    ) -> pyspark.sql.dataframe.DataFrame:
        """Completes missing values for a whole classified data as pyspark Dataframe.

        This method is similar to :py:meth:`~fill_and_fix_time_series_spark`,
        but here ``classification_data`` is a pyspark DataFrame containing
        a column called ``classification`` and the columns given by
        :py:attr:`~group_columns`, and it gives a classification
        to each time series uniquely  identified by ``self.group_columns``.
        The dictionary ``fill_dict`` must has a key for each unique
        classification found in the column ``classification``, and the
        value associated to the key is other dictionary with the arguments
        ``fillna_strategy``, ``duplicate_strategy``, ``outliers_detection``
        (Optional), ``interpolator`` (Optional) which are passed
        to the method :py:meth:`~fill_and_fix_time_series`.

        *This method tries to fix of fix each time series in a personalized
        way according to the categorization of ``classification``.

        """

        pyspark_validate_schema(
            data, self.target_column, self.time_column, *self.group_columns
        )

        self.__validate_fill_dict(
            classification_data=classification_data,
            fill_dict=fill_dict,
        )

        data_with_classification = data.join(
            classification_data, on=self.group_columns, how="inner"
        )
        # pylint: disable=unnecessary-lambda
        return data_with_classification.groupBy(
            self.group_columns
        ).applyInPandas(
            lambda d: self.fill_and_fix_time_series_from_grouped_data_on_classification(
                d,
                fill_dict=fill_dict,
                repeat_ffill_to=repeat_ffill_to,
                zero_bfill_to=zero_bfill_to,
                repeat_bfill_to=repeat_bfill_to,
            ),
            schema=self.__schema,
        )


# timeseries_helper = SparkTimeSeriesHelper(
#     target_column="y",
#     time_column="ds",
#     group_columns=["location_id", "item_id"],
#     freq="D",
# )

# data = 1

# data_daily = timeseries_helper.fill_and_fix_time_series_spark(
#     data=data,
#     duplicate_strategy="sum",
#     fillna_strategy="mean",
#     interpolator={"method": "linear", "limit": 10},
#     outliers_detection={"method": "iqr"},
#     zero_bfill_to="2021-01-01",
#     repeat_ffill_to="2022-10-01",
# )
