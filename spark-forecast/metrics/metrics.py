# -*- coding: utf-8 -*-
"""Metrics for model evaluation."""

from typing import Tuple

import numpy as np

from spark_forecast.exc import MetricDifferentLengthError

EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Simple error."""
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    if not np.any(_error(actual, predicted)) and not np.any(actual):
        return np.zeros(len(actual))
    if not np.any(actual) and np.any(predicted):
        return abs(_error(actual, predicted)) / (np.ones(len(actual)))
    _p_e = abs(_error(actual, predicted)) / (actual)
    _p_e[_p_e >= np.inf] = 1
    _p_e[np.isnan(_p_e)] = 0
    return _p_e


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1) -> np.ndarray:
    """Naive forecasting method which just repeats previous samples."""
    return actual[:-seasonality]


def _relative_error(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None
) -> np.ndarray:
    """Relative Error."""
    # TODO: strange parameter benchamark Is np.ndarray and then you ask if isinstance of int?
    # TODO: You ask twice isinstance(benchmark, int)
    # TODO: typing should be Optional[np.ndarray] for None defaults
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        seasonality = 1 if not isinstance(benchmark, int) else benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) / (
            _error(
                actual[seasonality:], _naive_forecasting(actual, seasonality)
            )
            + EPSILON
        )
    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)


def _bounded_relative_error(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None
) -> np.ndarray:
    """Bounded Relative Error."""
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        seasonality = 1 if not isinstance(benchmark, int) else benchmark
        abs_err = np.abs(_error(actual[seasonality:], predicted[seasonality:]))
        abs_err_bench = np.abs(
            _error(
                actual[seasonality:], _naive_forecasting(actual, seasonality)
            )
        )
    else:
        abs_err = np.abs(_error(actual, predicted))
        abs_err_bench = np.abs(_error(actual, benchmark))

    return abs_err / (abs_err + abs_err_bench + EPSILON)


def _geometric_mean(a) -> float:
    """Geometric mean."""
    new_a = 0
    if not np.any(a):
        return 0.0
    else:
        new_a = [i for i in a if i != 0]
    # Here we might face some problems when we recieve a 0 vector
    log_a = np.log(new_a)
    return np.exp(np.mean(log_a))


def wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted Mean Absolute Percent Error."""

    se_mape = abs(actual - predicted)

    return np.sum(se_mape) / (actual.sum() + EPSILON)


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Normalized Root Mean Squared Error."""
    if not np.any(actual):
        return 0.0
    return rmse(actual, predicted) / (actual.max() - actual.min())


def me(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Error."""
    return np.mean(_error(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(_error(actual, predicted)))


mad = mae  # Mean Absolute Deviation (it is the same as MAE)


def gmae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Geometric Mean Absolute Error."""

    return _geometric_mean(np.abs(_error(actual, predicted)))


def mdae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Median Absolute Error."""
    return np.median(np.abs(_error(actual, predicted)))


def mpe(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Percentage Error."""
    return np.mean(_percentage_error(actual, predicted))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def mdape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Median Absolute Percentage Error."""
    return np.median(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    return np.mean(
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + EPSILON)
    )


def smdape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Median Absolute Percentage Error."""
    return np.median(
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + EPSILON)
    )


def maape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Arctangent Absolute Percentage Error."""
    return np.mean(
        np.arctan(np.abs((actual - predicted) / (actual + EPSILON)))
    )


def mase(
    actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1
) -> float:
    """Mean Absolute Scaled Error Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)"""

    if mae(actual, predicted) == 0:
        return 0.0
    return mae(actual, predicted) / mae(
        actual[seasonality:], _naive_forecasting(actual, seasonality)
    )


def std_ae(actual: np.ndarray, predicted: np.ndarray):
    """Normalized Absolute Error."""
    __mae = mae(actual, predicted)
    return np.sqrt(
        np.sum(np.square(_error(actual, predicted) - __mae))
        / (len(actual) - 1)
    )


def std_ape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Normalized Absolute Percentage Error."""
    __mape = mape(actual, predicted)
    return np.sqrt(
        np.sum(np.square(_percentage_error(actual, predicted) - __mape))
        / (len(actual) - 1)
    )


def rmspe(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Percentage Error."""
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def rmdspe(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Median Squared Percentage Error."""
    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def rmsse(
    actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1
) -> float:
    """Root Mean Squared Scaled Error."""
    if not np.any(np.abs(_error(actual, predicted))):
        return 0.0
    elif (
        mae(actual[seasonality:], _naive_forecasting(actual, seasonality)) == 0
    ):
        return 0.0
    q = np.abs(_error(actual, predicted)) / mae(
        actual[seasonality:], _naive_forecasting(actual, seasonality)
    )
    return np.sqrt(np.mean(np.square(q)))


def inrse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Integral Normalized Root Squared Error."""

    return np.sqrt(
        np.sum(np.square(_error(actual, predicted)))
        / (np.sum(np.square(actual - np.mean(actual))) + EPSILON)
    )


def rrse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Relative Squared Error."""
    return np.sqrt(
        np.sum(np.square(actual - predicted))
        / (np.sum(np.square(actual - np.mean(actual))) + EPSILON)
    )


def mre(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None
) -> float:
    """Mean Relative Error."""
    return np.mean(_relative_error(actual, predicted, benchmark))


def rae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Relative Absolute Error (aka Approximation Error)"""
    return np.sum(np.abs(actual - predicted)) / (
        np.sum(np.abs(actual - np.mean(actual))) + EPSILON
    )


def mrae(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None
) -> float:
    """Mean Relative Absolute Error."""
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mdrae(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None
) -> float:
    """Median Relative Absolute Error."""
    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))


def gmrae(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None
) -> float:
    """Geometric Mean Relative Absolute Error."""
    return _geometric_mean(
        np.abs(_relative_error(actual, predicted, benchmark))
    )


def mbrae(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None
) -> float:
    """Mean Bounded Relative Absolute Error."""
    return np.mean(_bounded_relative_error(actual, predicted, benchmark))


def umbrae(
    actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None
) -> float:
    """Unscaled Mean Bounded Relative Absolute Error."""
    __mbrae = mbrae(actual, predicted, benchmark)
    return __mbrae / (1 - __mbrae)


def mda(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Directional Accuracy."""
    return np.mean(
        (
            np.sign(actual[1:] - actual[:-1])
            == np.sign(predicted[1:] - predicted[:-1])
        ).astype(int)
    )


METRICS = {
    "mse": mse,
    "rmse": rmse,
    "nrmse": nrmse,
    "me": me,
    "mae": mae,
    "mad": mad,
    "wmape": wmape,
    "gmae": gmae,
    "mdae": mdae,
    "mpe": mpe,
    "mape": mape,
    "mdape": mdape,
    "smape": smape,
    "smdape": smdape,
    "maape": maape,
    "mase": mase,
    "std_ae": std_ae,
    "std_ape": std_ape,
    "rmspe": rmspe,
    "rmdspe": rmdspe,
    "rmsse": rmsse,
    "inrse": inrse,
    "rrse": rrse,
    "mre": mre,
    "rae": rae,
    "mrae": mrae,
    "mdrae": mdrae,
    "gmrae": gmrae,
    "mbrae": mbrae,
    "umbrae": umbrae,
    "mda": mda,
}


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics: Tuple[str, ...] = ("mae", "mape", "mse", "rmse", "wmape"),
):

    if not set(metrics) <= METRICS.keys():
        raise ValueError(
            f"Las metricas disponibles son '{list(METRICS.keys())}'"
        )
    if not len(actual) == len(predicted):
        raise MetricDifferentLengthError(
            len_actual=len(actual), len_predicted=len(predicted)
        )

    return {name: METRICS[name](actual, predicted) for name in metrics}


def evaluate_all(actual: np.ndarray, predicted: np.ndarray):
    return evaluate(actual, predicted, tuple(METRICS.keys()))
