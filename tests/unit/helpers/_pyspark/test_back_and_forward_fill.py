# -*- coding: utf-8 -*-
"""Tests for fillers and interpolators on time series."""

import pandas as pd


from spark_forecast.helpers._pyspark.time_series import SparkTimeSeriesHelper
from spark_forecast.preprocessing.utils import (
    check_is_datetime_indexed_and_frequency,
)


def test_zero_backfill():
    serie = pd.Series(
        range(2, 12), pd.date_range("2022-01-01", periods=10, freq="H")
    )
    # zero_bfill kwarg has prioriry over
    serie_extended = SparkTimeSeriesHelper.fill_and_fix_time_series(
        serie,
        freq="H",
        fillna_strategy="median",
        duplicate_strategy="mode",
        zero_bfill_to="2021-12-31 22:00:00",
        repeat_bfill_to="2021-12-31 21:00:00",
    )

    check_is_datetime_indexed_and_frequency(serie_extended, "H")

    assert serie_extended.index.equals(
        pd.date_range("2021-12-31 22:00:00", periods=12, freq="H")
    )
    assert (
        serie_extended[
            pd.date_range("2021-12-31 22:00:00", periods=2, freq="H")
        ]
        == 0
    ).all()

    assert (
        serie_extended[pd.date_range("2022-01-01", periods=10, freq="H")]
        == serie
    ).all()


def test_repeat_backfill():

    serie = pd.Series(
        range(2, 12), pd.date_range("2022-01-01", periods=10, freq="D")
    )
    serie_extended = SparkTimeSeriesHelper.fill_and_fix_time_series(
        serie,
        freq="D",
        fillna_strategy="median",
        duplicate_strategy="mode",
        repeat_bfill_to="2021-12-20",
    )

    check_is_datetime_indexed_and_frequency(serie_extended, "D")

    assert serie_extended.index.equals(
        pd.date_range("2021-12-20", periods=22, freq="D")
    )
    assert (
        serie_extended[pd.date_range("2021-12-20", periods=12, freq="D")] == 2
    ).all()

    assert (
        serie_extended[pd.date_range("2022-01-01", periods=10, freq="D")]
        == serie
    ).all()


def test_repeat_forwardfill():

    serie = pd.Series(
        range(1, 21), pd.date_range("2022-01-01", periods=20, freq="D")
    )
    serie_extended = SparkTimeSeriesHelper.fill_and_fix_time_series(
        serie,
        freq="D",
        fillna_strategy="median",
        duplicate_strategy="mode",
        repeat_ffill_to="2022-02-28",
    )

    check_is_datetime_indexed_and_frequency(serie_extended, "D")

    assert serie_extended.index.equals(
        pd.date_range("2022-01-01", periods=59, freq="D")
    )
    assert (
        serie_extended[pd.date_range("2022-01-21", periods=39, freq="D")] == 20
    ).all()

    assert (
        serie_extended[pd.date_range("2022-01-01", periods=20, freq="D")]
        == serie
    ).all()


def test_fill_do_nothing():

    serie = pd.Series(
        range(1, 21), pd.date_range("2022-01-01", periods=20, freq="D")
    )
    serie_extended = SparkTimeSeriesHelper.fill_and_fix_time_series(
        serie,
        freq="D",
        fillna_strategy="median",
        duplicate_strategy="mode",
        repeat_ffill_to="2022-01-06",
        zero_bfill_to="2022-01-02",
    )

    check_is_datetime_indexed_and_frequency(serie_extended, "D")
    assert (serie_extended == serie).all()


def test_fill_short_serie():

    serie = pd.Series(1, index=pd.to_datetime(["2022-01-01"]))

    serie_extended = SparkTimeSeriesHelper.fill_and_fix_time_series(
        serie,
        freq="D",
        fillna_strategy="median",
        duplicate_strategy="mode",
        repeat_ffill_to="2022-01-06",
        zero_bfill_to="2021-12-30",
    )

    assert serie_extended.index.equals(
        pd.date_range("2021-12-30", "2022-01-06")
    )
