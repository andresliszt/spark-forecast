# -*- coding: utf-8 -*-
"""Naive demand classification."""

import enum

import pandas as pd


class DemandClassification(enum.Enum):
    """Types of time series classification."""

    zero = "Zero"
    """If all values are zeros"""
    lumpy = "Lumpy"
    """Demand with large quantity variation and time variation."""
    erratic = "Erratic"
    "Demand with regular time occurrences and considerable quantity variations."
    intermittent = "Intermittent"
    """Demand with large quantity variation and time variation."""
    smooth = "Smooth"
    """Easily forecastable by traditional methods"""


class NaiveDemandClassifier:
    """Simple Time Series Demand Classifier by bussiness rules:

    Original idea obtained from: Spiliotis, Evangelos & Makridakis,
    Spyros & Semenoglou, Artemios-Anargyros & Assimakopoulos, Vassilis. (2022).
    Comparison of statistical and machine learning methods for
    daily SKU demand forecasting. Operational Research.

    The main objective is to classify the demand by item in our time series
    dataset to verify how complex is to forecast the serie. Its based in two spectrums:

    1. The Average Demand Interval (ADI) — It measures the demand regularity in time by
    computing the average interval between two demands.

    2. The Square of the Coefficient of Variation (CV²) - It measures the variation
    in quantities.

    The classifications are:

    Smooth: Easily forecastable by traditional methods.

    Intermitent: Demand with a history of no pattern and very little variation in
    quantity but very high variation in demand intervals. Only a few forecasting
    methods can handle forecasting for intermittent demand.

    Erratic: Demand with regular time occurrences and considerable quantity variations.
    Forecasting this kind of demand is difficult.

    Lumpy: Demand with large quantity variation and time variation.
    Forecasting for Lumpy demand is not possible, no matter which
    forecasting method is getting used these demands are said to be unforecastable.

    Zero: A time serie which demand is zero everywhere and because of that
    is unforecastable.

    This method takes a pandas Series in which its index has to be a DateTime index type
    and the values must be the demand values for the item.

    Example
    -------
    >>> y = pd.Series(
        [1,10, 20,50,10, 13, 11.5],
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
        )
    >>> NaiveDemandClassifier().classify(y)
    'Intermitent'

    Returns
    -------
        The appropriate classification for the time series provided.

    """

    @staticmethod
    def coef_of_variance(y: pd.Series) -> float:
        """Calculates the Coefficient of Variation.

        The formula is given by :math:`\\frac{\\sigma}{\\mu^2}`

        """

        # TODO: What if mean = 0?
        return y.std() ** 2 / y.mean() ** 2

    @staticmethod
    def average_demand_interval(y: pd.Series) -> float:
        """Method to calculate the Average Demand Interval."""

        return (
            y.index - pd.to_datetime(pd.Series(y.index).shift(1))
        ).dt.days.mean()

    def classify(self, y: pd.Series) -> str:
        """Method to classify given the ``ADI`` and ``cv_sqr`` values for time serie."""

        if not isinstance(y.index, pd.core.indexes.datetimes.DatetimeIndex):
            raise TypeError(
                f"`y` must be time indexed `pd.core.indexes.datetimes.DatetimeIndex`, received {type(y.index)}"
            )

        if not isinstance(y, pd.Series):
            raise TypeError(
                f"`y` must be a `pandas.Series`, received {type(y)}"
            )

        if (y == 0).all():
            return DemandClassification.zero.value

        adi, cv_sqr = self.average_demand_interval(y), self.coef_of_variance(y)

        if (adi <= 1.34) and (cv_sqr <= 0.49):
            return DemandClassification.smooth.value
        if (adi >= 1.34) and (cv_sqr >= 0.49):
            return DemandClassification.lumpy.value
        if (adi < 1.34) and (cv_sqr > 0.49):
            return DemandClassification.erratic.value
        if (adi > 1.34) and (cv_sqr < 0.49):
            return DemandClassification.intermittent.value

        return DemandClassification.zero.value
