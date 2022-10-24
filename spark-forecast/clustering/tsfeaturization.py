from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from statsmodels.formula.api import ols
from statsmodels.sandbox.gam import AdditiveModel

from spark_forecast.clustering.utils.orthogonal_polynomial import (
    OrthogonalPolynomials,
)
from spark_forecast.exc import ColumnNamesError


class TSFeaturization:
    @staticmethod
    def run_length_encoding(y):
        """
        :param x: np.array
        :return: np.array
        """
        (pos,) = np.where(np.diff(y) != 0)
        pos = np.concatenate(([0], pos + 1, [len(y)]))
        rle = [b - a for (a, b) in zip(pos[:-1], pos[1:])]
        return rle

    @staticmethod
    def crossing_points(y):
        """
        :param y: np.array
        :return: _type_
        """
        ab = (y <= ((y.max() - y.min()) / 2)).values
        y_size = len(y)
        p1 = ab[1 : (y_size - 1)]
        p2 = ab[2:y_size]
        cross = (p1 & p2) | (p2 & ~p1)
        return cross.sum()

    @staticmethod
    def lumpiness(x, width):
        """Lumpiness."""
        nr = len(x)
        start = np.arange(1, nr, step=width, dtype=int)
        end = np.arange(width, nr + width, step=width, dtype=int)

        nsegs = int(nr / width)

        varx = np.zeros(nsegs)

        for idx in range(nsegs):
            tmp = x[start[idx] : end[idx]]
            varx[idx] = tmp[~np.isnan(tmp)].var()

        lump = varx[~np.isnan(varx)].var()
        return lump

    @staticmethod
    def trend_seasonality_spike_strength(y, freq):
        """Strength of trend and seasonality and spike."""
        y_size = len(y)
        season, peak, trough = (np.nan, np.nan, np.nan)

        if freq > 1:
            all_stl = sm.tsa.seasonal_decompose(y, period=freq)
            trend0 = all_stl.trend
            adj_x = y_size - (trend0 + all_stl.seasonal)
            detrend = y - trend0
            peak = all_stl.seasonal.max()
            trough = all_stl.seasonal.min()
            remainder = all_stl.resid
            season = (
                0
                if detrend.var() < 1e-10
                else max(0, min(1, 1 - adj_x.var() / detrend.var()))
            )

        else:  # No seasonal component
            tt = np.array([range(y_size)]).T

            _trend0_values = AdditiveModel(tt).fit(y.values).mu
            trend0 = pd.Series(_trend0_values, index=y.index)
            remainder = y - trend0
            deseason = y - trend0
            v_adj = trend0.var()

        trend = (
            0
            if deseason.var() < 1e-10
            else max(0, min(1, 1 - v_adj / deseason.var()))
        )

        n = len(remainder)
        v = remainder.var()
        d = (remainder - remainder.mean()) ** 2
        varloo = (v * (n - 1) - d) / (n - 2)
        spike = varloo.var()
        pl = OrthogonalPolynomials()
        pl.fit(range(y_size), degree=2)
        result_pl = pl.predict(range(y_size))

        X = sm.add_constant(result_pl, has_constant="add")
        ols_data = trend0.copy()
        ols_data = pd.concat(
            [ols_data.reset_index(drop=True), pd.DataFrame(X)],
            axis=1,
            ignore_index=True,
        )
        ols_data.columns = ["Y", "Intercept", "X1", "X2", "X3"]
        result_ols = ols("Y ~ X1 + X2 + X3", data=ols_data.dropna())

        trend_coef = result_ols.fit().params
        linearity = trend_coef[1]
        curvature = trend_coef[2]

        result = dict(
            trend=trend,
            spike=spike,
            peak=peak,
            trough=trough,
            linearity=linearity,
            curvature=curvature,
        )

        if freq > 1:
            result["season"] = season

        return result

    def featurization(
        self,
        data: pd.DataFrame,
        target_column: str,
        freq=1,
        width=None,
        window=None,
    ) -> pd.DataFrame:
        """_summary_

        :param y: _description_
        :type y: pd.Series

        """
        if target_column not in data.columns:
            raise ColumnNamesError(
                columns=list(data.columns), requested_columns=[target_column]
            )

        if data.shape[0] < (2 * freq):
            raise ValueError("Frequency given is to large for the time serie.")
        if width is None:
            width = freq if freq > 1 else 10

        if window is None:
            window = width

        if (width <= 1) | (window <= 1):
            raise ValueError("Window widths should be greater than 1.")

        y = data[target_column]
        features = dict()
        # features['entropy'] = spectral_entropy(y, sf=freq, method='welch', normalize=False)
        features["Lumpiness"] = self.lumpiness(y, width=width)
        features["ACF1"] = y.autocorr(1)
        features["Level Shift"] = (
            y.rolling(width).mean().diff(width).abs().max()
        )
        features["Variance Change"] = (
            y.rolling(width).var().diff(width).abs().max()
        )
        features["Crossing Median"] = self.crossing_points(y)
        features["Flat Spots"] = max(
            self.run_length_encoding(
                pd.cut(y, bins=10, include_lowest=True, labels=False)
            )
        )

        varts = self.trend_seasonality_spike_strength(y, freq=freq)
        features["Trend"] = varts["trend"]
        features["Linearity"] = varts["linearity"]
        features["Curvature"] = varts["curvature"]
        features["Spikeness"] = varts["spike"]

        return pd.DataFrame(features, index=[0])

    def clusterize(
        self,
        data: pd.DataFrame,
        group_columns: List[str],
        target_column: str = "y",
        **kwargs
    ) -> pd.DataFrame:
        """_summary_

        :param data: _description_
        :type data: pd.DataFrame
        :param group_columns: _description_
        :type group_columns: List[str]

        """
        grouped = (
            data[group_columns + list(target_column)]
            .groupby(group_columns)
            .apply(
                lambda y: self.featurization(y, target_column=target_column)
            )
        )
        normlized_grouping = (grouped - grouped.min()) / (
            grouped.max() - grouped.min()
        ).fillna(0).to_numpy()
        kmean = KMeans(**kwargs).fit(normlized_grouping)
        normlized_grouping["Cluster"] = kmean.labels_

        return normlized_grouping
