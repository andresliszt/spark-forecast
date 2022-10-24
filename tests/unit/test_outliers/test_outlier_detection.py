# import random
# import pandas as pd
# import pytest

# from spark_forecast.outlier_detection.outlier_detection import OUTLIERS

# TEST_DATA_ALL = [
#     (
#         pd.Series(
#             [random.randint(1, 3) for i in range(0, 99)] + [100],
#             index=pd.date_range("2022-01-01", periods=100),
#         ),
#         [-1, -1, -1],
#     ),
#     (
#         pd.Series(
#             [random.randint(1, 3) for i in range(0, 99)] + [10],
#             index=pd.date_range("2022-01-01", periods=100),
#         ),
#         [-1, -1, -1],
#     ),
#     (
#         pd.Series(
#             [random.randint(8, 10) for i in range(0, 99)] + [0],
#             index=pd.date_range("2022-01-01", periods=100),
#         ),
#         [-1, -1, 1],
#     ),
#     (
#         pd.Series(
#             [random.randint(7, 10) for i in range(0, 99)] + [-2],
#             index=pd.date_range("2022-01-01", periods=100),
#         ),
#         [-1, -1, 1],
#     ),
#     (
#         pd.Series(
#             [1, 3, 3, 2, 3, 5, 7, 3, 5, 9, 10, 3, 3, 3, 4, 11, 23, 20, 10]
#             + [3],
#             index=pd.date_range("2022-01-01", periods=20),
#         ),
#         [1, 1, 1],
#     ),
# ]


# @pytest.mark.parametrize(
#     "serie, expected_output",
#     TEST_DATA_ALL,
# )
# def test_univariate_outliers(serie, expected_output):

#     outliers_methods = [
#         OUTLIERS.iqr_outliers(serie)[-1],
#         OUTLIERS.modified_zscore(serie)[-1],
#         OUTLIERS.apply_rolling_window_outliers(serie)[-1],
#         #1 if OUTLIERS.dbescan_outliers(serie)[-1] != -1 else -1,
#         #OUTLIERS.isolation_forest_outliers(serie, random_state=1)[-1],
#     ]
#     assert outliers_methods == expected_output


# TEST_IMPORTANT_PARAMS = [
#     (
#         pd.Series(
#             [
#                 1,
#                 3,
#                 2,
#                 8,
#                 5,
#                 9,
#                 8,
#                 1,
#                 11,
#                 20,
#                 55,
#                 30,
#                 40,
#                 90,
#                 100,
#                 200,
#                 100,
#                 80,
#                 1,
#                 2,
#             ],
#             index=pd.date_range("2022-02-02", periods=20),
#         ),
#         0.5,
#         5,
#         0.5,
#         10,
#         9,
#         7,
#     ),
#     (
#         pd.Series(
#             [
#                 1,
#                 3,
#                 2,
#                 8,
#                 5,
#                 9,
#                 8,
#                 1,
#                 11,
#                 20,
#                 55,
#                 30,
#                 40,
#                 90,
#                 100,
#                 200,
#                 100,
#                 80,
#                 1,
#                 2,
#             ],
#             index=pd.date_range("2022-02-02", periods=20),
#         ),
#         0.4,
#         10,
#         0.6,
#         8,
#         8,
#         7,
#     ),
#     (
#         pd.Series(
#             [
#                 1,
#                 3,
#                 2,
#                 8,
#                 5,
#                 9,
#                 8,
#                 1,
#                 11,
#                 20,
#                 55,
#                 30,
#                 40,
#                 90,
#                 100,
#                 200,
#                 100,
#                 80,
#                 1,
#                 2,
#             ],
#             index=pd.date_range("2022-02-02", periods=20),
#         ),
#         0.2,
#         50,
#         0.8,
#         4,
#         1,
#         5,
#     ),
#     (
#         pd.Series(
#             [
#                 1,
#                 3,
#                 2,
#                 8,
#                 5,
#                 9,
#                 8,
#                 1,
#                 11,
#                 20,
#                 55,
#                 30,
#                 40,
#                 90,
#                 100,
#                 200,
#                 100,
#                 80,
#                 1,
#                 2,
#             ],
#             index=pd.date_range("2022-02-02", periods=20),
#         ),
#         0.1,
#         100,
#         0.9,
#         2,
#         0,
#         5,
#     ),
# ]


# @pytest.mark.parametrize(
#     "serie, contamination, eps, quantile, expected_amount_if, expected_amount_dbscan, expected_rolling_quantile",
#     TEST_IMPORTANT_PARAMS,
# )
# def test_parameters_outliers(
#     serie,
#     contamination,
#     eps,
#     quantile,
#     expected_amount_if,
#     expected_amount_dbscan,
#     expected_rolling_quantile,
# ):
#     #iforest = OUTLIERS.isolation_forest_outliers(
#     #    serie, contamination=contamination, random_state=1
#     #)
#     #dbscan = OUTLIERS.dbescan_outliers(serie, eps=eps)
#     rq = OUTLIERS.apply_rolling_window_outliers(serie, quantile=quantile)

#     #assert len(iforest[iforest.values == -1]) == expected_amount_if
#     #assert len(dbscan[dbscan == -1]) == expected_amount_dbscan
#     assert len(rq[rq.values == -1]) == expected_rolling_quantile
