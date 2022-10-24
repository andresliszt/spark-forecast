# -*- coding: utf-8 -*-
"""pandas API integration tests."""


from spark_forecast.helpers._pyspark.time_series import SparkTimeSeriesHelper
from spark_forecast.classification.demand_classifier import (
    DemandClassification,
)


def test_complete_dates_in_time_series_in_pandas_pyspark(
    pyspark_test_data,
):

    helper = SparkTimeSeriesHelper(
        freq="D",
        target_column="y",
        time_column="ds",
        group_columns=["location_id", "item_id"],
    )

    pyspark_test_data = helper.fill_and_fix_time_series_spark(
        pyspark_test_data, duplicate_strategy="mode", fillna_strategy="zero"
    )
    classification = helper.naive_time_series_classifier_spark(
        pyspark_test_data
    )

    classification = classification.toPandas()

    assert set(classification.classification.unique()) <= set(
        [e.value for e in DemandClassification]
    )
