# -*- coding: utf-8 -*-
"""Outliers detection/transformation method."""

from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from spark_forecast.preprocessing.utils import check_time_series_not_null


class OutlierProcessing:
    """Outlier Detection Methods:

    This class recives a Pandas Series or Pandas Datframes to which will apply
    the chosen method by user. The methods available are:

    1. Interquantile Range(iqr_outliers(y))
    2. Modified ZScore(modified_zscore(y))
    3. Rolling Quantile(apply_rolling_window_outliers(y))
    4. DBScan(dbescan_outliers(y))
    5. Isolation Forest(isolation_forest_outliers(y))

    Example
    -------
    >>> y = pd.Series(
        [1, 5, 1, 2, 3, 100, 80, 4, 2, 2],
        index=pd.date_range("2022-01-01", periods=10),
    )
    >>> OutlierProcessing().isolation_forest_outliers(y, contamination = .2)

    2022-01-01    1
    2022-01-02    1
    2022-01-03    1
    2022-01-04    1
    2022-01-05    1
    2022-01-06   -1
    2022-01-07   -1
    2022-01-08    1
    2022-01-09    1
    2022-01-10    1
    Freq: D, Name: outliers, dtype: int32

    Returns
    -------
        Pandas Series with values 1 or -1 where -1 represents the outliers
    in the inputed serie.

    """

    @staticmethod
    def __true_false_to_minus_one(outliers_mask: pd.Series) -> pd.Series:
        """Method that replace true and false for 1 and -1 respectively.

        :param outliers_mask: Pandas Series with True and False as values.

        :return: Pandas Series with values 1 and -1 as a replacement for
        True and False values of the serie received as input.

        """
        return outliers_mask.replace({True: 1, False: -1})

    @staticmethod
    def __sckit_learn_data_input(
        data: Union[pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:

        return (
            data.values.reshape(-1, 1) if isinstance(data, pd.Series) else data
        )

    @staticmethod
    def stationarity_test(
        X: Union[pd.DataFrame, pd.Series], column: Optional[str]
    ) -> str:
        """_summary_

        :param X: _description_
        :type X: Union[pd.DataFrame, pd.Series]
        :param column: _description_, defaults to ''
        :type column: str, optional
        :return: _description_
        :rtype: str

        """
        adf_test = (
            adfuller(X, autolag="AIC")
            if isinstance(X, pd.Series)
            else adfuller(X[column], autolag="AIC")
        )
        if adf_test[1] <= 0.05:
            return True
        return False

    @staticmethod
    def var_calcs(X: Union[pd.DataFrame, pd.Series], threshold_multiplier):
        var = VAR(X)
        lags = var.select_order().aic
        var_fitresults = var.fit(lags)
        squared_errors = var_fitresults.resid.sum(axis=1) ** 2
        threshold = np.mean(squared_errors) + threshold_multiplier * np.std(
            squared_errors
        )
        return squared_errors >= threshold

    def iqr_outliers(self, y: pd.Series) -> pd.Series:
        """InterQuartile Range method to identify outliers: sets up a “fence” outside of Q1 and Q3. Any values that fall outside of this fence are considered outliers.

        :param y: Pandas Series that contains the target as values
        to analyse through the method IQR

        :return: Pandas Series with values 1 and -1 where 1 represents
        a "normal" value and -1 represents an outlier of the inputed serie.

        """
        # Series can´t has null/nan values
        check_time_series_not_null(y)
        quartile_1, quartile_3 = np.percentile(y, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 2.2)
        upper_bound = quartile_3 + (iqr * 2.2)
        outliers_mask = (y >= lower_bound) & (y <= upper_bound)
        outliers_mask.name = "outliers"
        return self.__true_false_to_minus_one(outliers_mask)

    def modified_zscore(self, y: pd.Series, thres=5.0) -> pd.Series:
        """Statisitical method for measuring deviation from MAD.

        :param y: Pandas Series that contains the target to
        analyse through the method Modified Zscore.

        :param thres: Threshold value is used to compute the
        bounds to identify wheter a value is an outlier or not.

        :return: Pandas Series with values 1 and -1 where 1
        represents a "normal" value and -1 represents an outlier
        of the inputed serie.

        """
        # Series can´t has null/nan values
        check_time_series_not_null(y)
        median_absolute_deviation_y = (y - y.median()).abs().median()

        if median_absolute_deviation_y != 0:
            diff = y - y.median()
            modified_z_scores = 0.6745 * diff / median_absolute_deviation_y
        else:
            modified_z_scores = stats.zscore(y)
        outliers_mask = np.isnan(modified_z_scores) | (
            np.abs(modified_z_scores) < thres
        )
        outliers_mask.name = "outliers"
        return self.__true_false_to_minus_one(outliers_mask)

    def apply_rolling_window_outliers(
        self, y: pd.Series, quantile=0.90
    ) -> pd.Series:
        """Rolling Window method of outlier capping.

        Using a certain window of days (Two weeks by default) and a
        certain threshold based on quantiles (0.95 by default) any value
        greater than the threhold is capped to the threshold value.

        :param y: Pandas Series that contains the target to
        analyse through the method rolling quantiles.

        :param quantile: quantile value as threshold to determie wheter
        a value is outlier.

        :return: Pandas Series with values 1 and -1 where 1
        represents a "normal" value and -1 represents an outlier
        of the inputed serie.

        """
        # Series can´t has null/nan values
        check_time_series_not_null(y)
        # SMA Calculations
        sma_serie = y.rolling(window=int(len(y) / 2)).mean().fillna(y.mean())
        # SMA Difference
        sma_diff_serie = np.absolute(y.to_numpy() - sma_serie.to_numpy())
        # Trimmed quantile
        v_max = np.quantile(sma_diff_serie, q=quantile)
        outliers_mask = y <= v_max
        outliers_mask.name = "outliers"
        return self.__true_false_to_minus_one(outliers_mask)

    def dbescan_outliers(
        self, data: Union[pd.Series, pd.DataFrame], **kwargs
    ) -> pd.Series:
        """Clustering method for Outlier detection.

        Non-parametric method. The clusters are defined on neighbouring points
        within a certain threhold of minimum points to be considered collective
        cluster.

        :param data: Pandas Series or Dataframe which will be clusterized

        :param eps: The maximum distance between two samples for one to
        be considered as in the neighborhood of the other. This is not a
        maximum bound on the distances of points within a cluster. This is
        the most important DBSCAN parameter to choose appropriately for
        your data set and distance function.

        :param min_samples: int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

        :param metric: str, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a :term:`Glossary <sparse graph>`, in which
        case only "nonzero" elements may be considered neighbors for DBSCAN.

        :param metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        :param algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

        :param leaf_size: Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.

        :param p: The power of the Minkowski metric to be used to calculate distance
        between points. If None, then ``p=2`` (equivalent to the Euclidean
        distance).

        :param n_jobs: int, default=None
        The number of parallel jobs to run.

        :return: Pandas Series with values between [-1, number of neighborhoods]
        where -1 represents the values considered as outliers.

        """
        # We only ignore n_jobs kwargument
        kwargs["n_jobs"] = None
        dbscan = DBSCAN(**kwargs)
        pred = pd.Series(
            dbscan.fit_predict(self.__sckit_learn_data_input(data)),
            index=data.index,
            name="outliers",
        )
        return pred.replace({0: 1, 1: 1})

    def isolation_forest_outliers(
        self, data: Union[pd.Series, pd.DataFrame], **kwargs
    ) -> pd.Series:
        """Unsupervised method for Outlier detection.

        Based on how difficult (or easy) is to isolate a point using tree-like space
        partitioning (Much like those of decision trees). The more branches we need to
        isolate a point, the less likely it is for that point to be an outlier.

        :param y: Pandas Series or Pandas DataFrame which will be analysed
        through this method

        :param n_estimators: The number of base estimators in the ensemble.

        :param max_samples: The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

        :param contamination: The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.

            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].

        :param max_features: The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

        :param bootstrap: If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

        :param n_jobs: The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

        :param random_state: Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

        :param verbose: Controls the verbosity of the tree building process.

        :param warm_start: When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

        :return: Pandas Series with values 1 and -1 where 1
        represents a "normal" value and -1 represents an outlier
        of the inputed serie.

        """
        # We only ignore n_jobs kwargument
        kwargs["n_jobs"] = None
        iso = IsolationForest(**kwargs)
        pred = pd.Series(
            iso.fit_predict(self.__sckit_learn_data_input(data)),
            index=data.index,
            name="outliers",
        )
        return pred

    def VAR_outliers(
        self, X: Union[pd.DataFrame, pd.Series], threshold_multipler: int = 1
    ) -> pd.Series:

        # adf_test_results = {col: self.stationarity_test(X, col) for col in X.columns} if isinstance(X, pd.DataFrame) else
        X_copy = X.copy()
        if isinstance(X, pd.DataFrame):
            for col in X_copy.columns:
                order = 0
                while self.stationarity_test(X_copy, col) and order < 4:
                    order += 1
                    X_copy[col] = (
                        X_copy[col].diff(order).fillna(X_copy[col].mean())
                    )

            predictions = self.var_calcs(X_copy, threshold_multipler)
            return self.__true_false_to_minus_one(
                pd.Series(
                    [
                        np.nan
                        for i in range(0, (X.shape[0] - predictions.shape[0]))
                    ]
                    + list(predictions.values),
                    index=X_copy.index,
                )
            )
        order = 0
        while self.stationarity_test(X_copy, None) and order < 4:
            order += 1
            X_copy = X_copy.diff(order).fillna(X_copy.mean())
        predictions = self.var_calcs(X_copy, threshold_multipler)
        return self.__true_false_to_minus_one(
            pd.Series(
                [np.nan for i in range(0, (X.shape[0] - predictions.shape[0]))]
                + list(predictions.values),
                index=X_copy.index,
            )
        )


# TODO: Multivariate detection must be refactorized, for now we are not including it

OUTLIERS = OutlierProcessing()

iqr_outliers = OUTLIERS.iqr_outliers
modified_zscore_outliers = OUTLIERS.modified_zscore
rolling_window_outliers = OUTLIERS.apply_rolling_window_outliers
# dbescan_outliers = OUTLIERS.dbescan_outliers
# isolation_outliers = OUTLIERS.isolation_forest_outliers

OUTLIERS_METHODS = {
    "iqr": iqr_outliers,
    "modified_zscore": modified_zscore_outliers,
    "rolling_window": rolling_window_outliers,
}


__all__ = (
    "OUTLIERS_METHODS",
    "iqr_outliers",
    "modified_zscore_outliers",
    "rolling_window_outliers",
)
