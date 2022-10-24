# -*- coding: utf-8 -*-
"""Outliers transformation method."""

from typing import List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

# pylint: disable=invalid-name, missing-class-docstring


class OutlierTransformer(TransformerMixin):
    """Class used to modifiy values for a given column list."""

    def __init__(self, outlier_columns: List[float]) -> None:

        self.outlier_columns = outlier_columns

    @staticmethod
    def grubbs_test(df_filtered_by_column: pd.Series) -> bool:
        """Statistic test for major outliers. The test decides whether to reject or fail to reject the null hypothesis H0: The dataset doesn't contain an outlier.

        WARNING: Grubb's test depends on normality assumptions.

        Do not apply when the number of points in the time series is less than 6.

        :param df_filtered_by_column: DataFrame's column which
            will be modified.

        """
        n = len(df_filtered_by_column)
        mean_x = df_filtered_by_column.mean()
        sd_x = df_filtered_by_column.std()
        numerator = np.max(np.abs(df_filtered_by_column - mean_x))
        g_calculated = numerator / sd_x
        t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
        g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (
            np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value))
        )
        return not g_critical > g_calculated

    @staticmethod
    def dbescan_outliers(
        df_filtered_by_column: pd.Series, eps=2.0
    ) -> pd.Series:
        """Clustering method for Outlier detection.

        Non-parametric method. The clusters are defined on neighbouring points
        within a certain threhold of minimum points to be considered collective
        cluster.

        :param df_filtered_by_column: DataFrame's column which
            will be clusterized.

        """
        outlier_detection = DBSCAN(eps=eps, metric="euclidean", min_samples=5)
        clusters = outlier_detection.fit_predict(
            df_filtered_by_column.values.reshape(-1, 1)
        )
        data = pd.Series()
        data["cluster"] = clusters
        return data.loc[data.cluster == -1].index

    @staticmethod
    def isolation_forest_outliers(
        df_filtered_by_column: pd.Series,
    ) -> pd.Series:
        """Unsupervised method for Outlier detection.

        Based on how difficult (or easy) is to isolate a point using tree-like space
        partitioning (Much like those of decision trees). The more branches we need to
        isolate a point, the less likely it is for that point to be an outlier.

        :param df_filtered_by_column: DataFrame's column which
            will be modified.

        """
        iso = IsolationForest(random_state=1, contamination="auto")
        preds = iso.fit_predict(df_filtered_by_column.values.reshape(-1, 1))
        data = pd.Series()
        data["cluster"] = preds
        return data.loc[data.cluster == -1].index

    @staticmethod
    def modified_zscore(df_filtered_by_column: pd.Series) -> pd.Series:
        """Statisitical method for measuring deviation from MAD.

        :param df_filtered_by_column: DataFrame's column which
            will be clusterized.

        """
        thres = 5.0
        median_y = df_filtered_by_column.median()
        median_absolute_deviation_y = (
            (df_filtered_by_column - df_filtered_by_column.median())
            .abs()
            .median()
        )

        if median_absolute_deviation_y != 0:
            diff = df_filtered_by_column - median_y
            modified_z_scores = 0.6745 * diff / median_absolute_deviation_y
        else:
            modified_z_scores = stats.zscore(df_filtered_by_column)
        condition1 = np.isnan(modified_z_scores)
        condition2 = np.abs(modified_z_scores) < thres
        condition = condition1 | condition2
        return condition

    @staticmethod
    def apply_rolling_window_outliers(
        block_df, window=7 * 2, quantile=0.95, column="y", new_col="y_new"
    ) -> pd.DataFrame:
        """Rolling Window method of outlier capping.

        Using a certain window of days (Two weeks by default) and a
        certain threshold based on quantiles (0.95 by default) any value
        greater than the threhold is capped to the threshold value.

        """
        result_df = block_df.copy().fillna(0).sort_values("ds")
        mean_val = result_df[column].mean()
        result_df["SMA"] = (
            result_df[column].rolling(window=window).mean().fillna(mean_val)
        )
        result_df["SMA_diff"] = np.absolute(
            result_df[column].to_numpy() - result_df["SMA"].to_numpy()
        )
        v_max = np.quantile(result_df["SMA_diff"].to_numpy(), q=quantile)
        result_df[new_col] = result_df.y.apply(lambda x: np.clip(x, 0, v_max))
        result_df.loc[
            result_df[column] != result_df[new_col], "actual_amt"
        ] = (
            result_df.loc[
                result_df[column] != result_df[new_col], "actual_amt"
            ]
            * result_df.loc[result_df[column] != result_df[new_col], new_col]
            / result_df.loc[result_df[column] != result_df[new_col], column]
        )
        return result_df

    def fit(self, X: pd.DataFrame):  # pylint: disable=unused-argument
        return self

    def transform(self, X: pd.DataFrame):
        if X.empty:
            raise ValueError("'X' cant be empty")
        X_copy = X.copy()
        X_copy = X_copy.reset_index()

        for col in self.outlier_columns:
            x = list(self.modified_zscore(X_copy[col]))
            list_mod = [i for i, j in enumerate(x) if not j]

            if self.grubbs_test(X_copy[col]):
                indx_db = self.dbescan_outliers(X_copy[col])
                indx_if = self.isolation_forest_outliers(X_copy[col])
                indx_survivor = list(set(indx_db) & set(indx_if))

                if len(indx_survivor) > len(list_mod):
                    X_copy.loc[indx_survivor, col] = (
                        self.apply_rolling_window_outliers(X_copy)
                        .iloc[indx_survivor]
                        .y_new
                    )
                    return X_copy

                X_copy.loc[list_mod, col] = (
                    self.apply_rolling_window_outliers(X_copy)
                    .iloc[list_mod]
                    .y_new
                )
                return X_copy
            X_copy.loc[list_mod, col] = (
                self.apply_rolling_window_outliers(X_copy).iloc[list_mod].y_new
            )
            return X_copy
