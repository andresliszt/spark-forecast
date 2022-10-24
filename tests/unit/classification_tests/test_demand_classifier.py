# -*- coding: utf-8 -*-
"""
Testing Demand Classifier
"""

import pandas as pd
import pytest

from spark_forecast.classification.demand_classifier import (
    NaiveDemandClassifier,
)

TEST_DATA = [
    (
        pd.Series(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            index=pd.date_range("2022-02-10", "2022-02-19"),
        ),
        "Smooth",
    ),
    (
        pd.Series(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            index=pd.date_range("2022-02-10", "2022-02-19"),
        ),
        "Zero",
    ),
    (
        pd.Series(
            [1, 1, 1, 1, 1, 1],
            index=pd.to_datetime(
                [
                    "2022-02-10",
                    "2022-02-11",
                    "2022-02-14",
                    "2022-02-15",
                    "2022-02-24",
                    "2022-02-28",
                ]
            ),
        ),
        "Intermittent",
    ),
    (
        pd.Series(
            [0, 0, 20, 0, 0, 0, 0],
            index=pd.to_datetime(
                [
                    "2022-02-10",
                    "2022-02-11",
                    "2022-02-12",
                    "2022-02-15",
                    "2022-02-16",
                    "2022-02-17",
                    "2022-02-19",
                ]
            ),
        ),
        "Lumpy",
    ),
    (
        pd.Series(
            [0, 0, 20, 0, 0, 0, 0],
            pd.to_datetime(
                [
                    "2022-02-10",
                    "2022-02-11",
                    "2022-02-12",
                    "2022-02-15",
                    "2022-02-16",
                    "2022-02-17",
                    "2022-02-18",
                ]
            ),
        ),
        "Erratic",
    ),
]


@pytest.mark.parametrize("test_df, output_df", TEST_DATA)
def test_naive_demand_classifier(test_df, output_df):
    classifier = NaiveDemandClassifier().classify(test_df)
    assert classifier == output_df
