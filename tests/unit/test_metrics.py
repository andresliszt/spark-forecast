# -*- coding: utf-8 -*-
"""
Testing Metrics for Movel Evaluation
"""

import numpy as np
import pytest

from spark_forecast.metrics.metrics import METRICS
from spark_forecast.metrics.metrics import evaluate_all


class ExpectedMetrics:
    def __init__(self, **kwargs):
        if kwargs.keys() != METRICS.keys():
            raise RuntimeError
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {k: getattr(self, k) for k in METRICS}


TEST_DATA = [
    (
        np.zeros(10),
        np.zeros(10),
        (
            ExpectedMetrics(
                wmape=0.0,
                mse=0.0,
                rmse=0.0,
                nrmse=0.0,
                me=0.0,
                mae=0.0,
                mad=0.0,
                gmae=0.0,
                mdae=0.0,
                mpe=0.0,
                mape=0.0,
                mdape=0.0,
                smape=0.0,
                smdape=0.0,
                maape=0.0,
                mase=0.0,
                std_ae=0.0,
                std_ape=0.0,
                rmspe=0.0,
                rmdspe=0.0,
                rmsse=0.0,
                inrse=0.0,
                rrse=0.0,
                mre=0.0,
                rae=0.0,
                mrae=0.0,
                mdrae=0.0,
                gmrae=0.0,
                mbrae=0.0,
                umbrae=0.0,
                mda=1.0,
            )
        ),
    )
]


@pytest.mark.parametrize("y_actual,y_pred,expected_metrics", TEST_DATA)
def test_metrics(y_actual, y_pred, expected_metrics):
    assert (  # nosec
        evaluate_all(y_actual, y_pred) == expected_metrics.to_dict()
    )
