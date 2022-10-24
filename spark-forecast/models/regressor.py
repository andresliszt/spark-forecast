# -*- coding: utf-8 -*-
"""Models based on regression."""

import inspect
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from sklearn.utils import resample

from spark_forecast.exc import CovariatesWithNullValues
from spark_forecast.exc import CustomPredictorsWithNullValues
from spark_forecast.exc import RegressorNotFittedError
from spark_forecast.exc import TimeSeriesAndCovariatesIndexError
from spark_forecast.exc import TimeSeriesWithNullValues
from spark_forecast.exc import TooShortTimeSeries
from spark_forecast.helpers import logger
from spark_forecast.preprocessing.utils import (
    check_is_datetime_indexed_and_frequency,
)

from spark_forecast.models.local_forecasting import PREDICTIONS_COL_NAME


class RegressorForecast:
    """Forecast bassed on Machine Learning Regression techniques.

    This approach consists in build a matrix ``X_train`` based
    on lags of a time series ``y`` plus exogeneous variables
    that accross this project are called ``future_covariates``.
    This Regression ML forecast needs to know the ``future_covariates``
    among the whole time horizont i.e, as for training and as
    for predict.

    Multiples validations are performed to ensure that
    the time series being forecasted is *suitable*. In this
    class and the whole project, time index and defined
    frequency are mandatory.

    """

    def __init__(
        self,
        regressor,
        lags: int,
        freq: str,
    ) -> None:

        if not isinstance(lags, int) or lags < 1:
            raise ValueError("`lags` must be positive integer")

        if inspect.isclass(regressor):
            raise ValueError(
                "`regressor` must be an instance of a sckit-learn class."
                "Example: `RandomForestRegressor()` instead of `RandomForestRegressor`."
            )
        # Number of lags
        self.lags = lags
        # Frequency compatible with pandas notation (e.g "D")
        self.freq = freq
        # Any regressor with fit and predit methods
        self.regressor = regressor
        # status of the regressor
        self.fitted = False
        # If model included covariates
        self.trained_with_covariates = False
        # names of the columns of the covariates DF
        self.covariates_names = None
        # In training time saved to predict T+1
        self.last_window = None
        # In training time saved as the last date found in the serie
        self.last_date = None
        # DF of shape (n_steps, lags + covariates) with predictions
        # and features of that predictions
        self.observations_and_predictions = None
        # Names of the lags columns for DF
        self.lags_columns = [f"lag_{n}" for n in range(1, self.lags + 1)]

    @staticmethod
    def __check_covariates(
        future_covariates: Union[pd.Series, pd.DataFrame]
    ) -> None:
        """Check covariates for train.

        Must be pandas Series or DataFrame and not contain null values.

        """

        if not isinstance(future_covariates, (pd.Series, pd.DataFrame)):
            raise TypeError(
                f"`future_covariates` must be a pandas Series or pandas \
                    Dataframe. Got {type(future_covariates)}"
            )
        if (
            isinstance(future_covariates, pd.Series)
            and future_covariates.isnull().any()
        ) or future_covariates.isnull().values.any():
            raise CovariatesWithNullValues

    def __check_covariates_for_predict(
        self, future_covariates: Union[pd.Series, pd.DataFrame], steps: int
    ) -> None:
        """Check covariates for predict.

        The number of rows of ``future_covariates``` must be ``steps``
        (i.e the number of predictions you want to make). Check if the
        model was trained with covariates checking bool value of
        ``self.trained_with_covariates``. Also, check if the features
        names found at training match with the names in
        ``future_covariates``.

        """
        self.__check_covariates(future_covariates)

        if not self.trained_with_covariates:
            raise ValueError(
                "Regressor wasn't trained with \
                covariates and predict got `future_covariates`"
            )

        if steps != len(future_covariates):
            raise ValueError(
                "Lenght of the `future_covariates` must be \
                    the number of steps required"
            )

        cols = (
            list(future_covariates.columns)
            if isinstance(future_covariates, pd.DataFrame)
            else self.covariates_names
        )

        if cols != self.covariates_names:
            raise ValueError(
                "Covariates has different column names from \
                        the ones that were trained"
            )

    def __check_y(self, y: pd.Series) -> None:
        """Check time series to be forecasted.

        ``y`` must be a pandas Series with time index and with frequency
        equals to ``self.freq``, and not contain null values.

        """
        if not isinstance(y, pd.Series):
            raise TypeError(f"`y` must be a pandas Series. Got {type(y)}")

        check_is_datetime_indexed_and_frequency(y, self.freq)
        if y.isnull().any():
            raise TimeSeriesWithNullValues

        if len(y) < self.min_train_series_length:
            logger.error(
                "Serie must be larger than lags size",
                lags=self.lags,
            )
            raise TooShortTimeSeries(
                length_received=len(y),
                length_minimum=self.min_train_series_length,
            )

    @staticmethod
    def __check_covariates_and_series_indexes(y, future_covariates):
        if not y.index.equals(future_covariates.index):
            raise TimeSeriesAndCovariatesIndexError

    def _is_probabilistic(self) -> bool:
        """Needed method for ``darts``"""
        return True

    @property
    def min_train_series_length(self):
        return self.lags + 1

    def create_lags(
        self,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Create lags from ``y``.

        Builds a pandas Data Frame of shape
        (``len(y) - self.lags``, ``self.lags``) that will be part
        of the training matrix ``X_train`` for :py:meth:`~fit`.
        The row number :math:`t` represent consist in the observations
        :math:`y_{t-1}, y_{t-2}, ... , y_{t - \text{lags}}` used
        to predcit :math:`y_{t}`.

        """

        self.__check_y(y)

        lags_features = pd.concat(
            [y.shift(lg) for lg in range(1, self.lags + 1)], axis=1
        )
        lags_features.columns = self.lags_columns
        # lags_features has dtype float because of shift method
        return lags_features[-(len(y) - self.lags) :]

    def create_train_set(
        self,
        y: pd.Series,
        future_covariates: Optional[Union[pd.Series, pd.DataFrame]],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Creates training set to feed regressor.

        Builds matrix ``X_train`` and target values ``y_train``
        that are passed to train :py:attr:`~regressor`. ``X_train``
        is made of the lags calling :py:meth:`~create_lags` and,
        if provided, exogeneous variables ``future_covariates``.

        Multiples validations are made on ``y`` and ``future_covariates``,
        they must have the same time index, and be numeric type
        (for example, boolean values will be cast to 0 and 1) and
        can't contain null values.

        Note that ``X_train`` has ``len(y) - lags`` rows and
        ``y_train`` is only ``y[lags:]``.

        Parameters
        ----------
        y
            Time series to be forecasted.
        future_covariates
            If provided, exogenous variables.

        Returns
        -------
            ``X_train`` and ``y_train`` to feed regressor.

        """

        y = y.astype(float)
        # We create lags features
        lags_features = self.create_lags(y)

        if future_covariates is not None:
            self.__check_covariates(future_covariates)
            future_covariates = future_covariates.astype(float)

            # We check that covariates has right index
            self.__check_covariates_and_series_indexes(y, future_covariates)
            self.trained_with_covariates = True

            return (
                pd.concat(
                    [
                        lags_features,
                        future_covariates.loc[lags_features.index],
                    ],
                    axis=1,
                ),
                y[lags_features.index],
            )
        # We return X, y as a numpy np.ndarray
        return lags_features, y[lags_features.index]

    def fit(
        self,
        y,
        future_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ) -> None:
        """Fits regressor and save fit status.

        It calls :py:meth:`~create_train_set` and then
        fit the :py:attr:`~regressor`. Also this method
        saves the last date found on ``y``, that will be the
        start point to predict the next step time and the last window
        needed to predict it.


        Parameters
        ----------
        y
            Time series to be forecasted.
        future_covariates
            If provided, exogenous variables.

        """
        X_train, y_train = self.create_train_set(
            y, future_covariates=future_covariates
        )
        if future_covariates is not None:
            self.covariates_names = (
                future_covariates.name
                if isinstance(future_covariates, pd.Series)
                else list(future_covariates.columns)
            )

        # TODO: When include transformers on exog and y, we must remove .values
        # We fit regressor with numpy ndarrays
        self.regressor.fit(X_train.values, y_train.values)
        # Set fitted state as True
        self.fitted = True
        # We save last_window as a DataFrame of one row
        # This is because we are fitting with X_train as a DataFrame
        # If we want to predict, sckit-learn will prefer a DataFrame (not mandatory, but better)
        # Last window will be the start point for recursive predict
        self.last_window = np.flip(np.hstack(y[-self.lags :]))
        # Last date will be start point for recursive predict
        self.last_date = y.index[-1]

    def _recursive_predict(self, steps: int) -> pd.Series:
        """Recursive predict for multistep forecast.

        When :py:meth:`~fit` is called, the attribute
        :py:attr:`~last_window` and :py:attr:`~last_date`
        are saved with the values found in the time
        series are the starting points to multi step
        predictions. The last window is used to predict
        the next time given increasing the last date
        in one unit of time by :py:attr:`~freq` and in
        an iterative way the last window is updated using
        the value of the current prediction.

        Also we save the values of the observation needed
        to make the predictions with purpose of use it
        in :py:meth:`~predict_interval` method


        Parameters
        ----------
        steps
            Number of predictions.

        Returns
        -------
            Time series with the predictions indexed by
            time, starting from :py:attr:`~last_date`

        """

        observations_and_predictions = pd.DataFrame(
            np.nan,
            columns=[
                PREDICTIONS_COL_NAME,
                *self.lags_columns,
            ],
            index=pd.date_range(
                self.last_date, periods=steps + 1, freq=self.freq
            )[1:],
        )
        # Last window
        last_window = self.last_window

        for stp in range(steps):
            # pred is a float with the prediction of the next step
            pred = self.regressor.predict(last_window.reshape(1, -1))[0]
            # observations_and_predictions we save all predictions and observation
            observations_and_predictions.iloc[stp] = [
                pred,
                *last_window,
            ]
            # We update value of last_window for the next iteration
            last_window = np.array([pred, *last_window[:-1]])

        # We use the same dtype as last_window
        # We save observation and predictions for boostrap interval predict
        self.observations_and_predictions = observations_and_predictions

        return observations_and_predictions[PREDICTIONS_COL_NAME]

    def _recursive_predict_with_covariates(
        self, future_covariates: Union[pd.Series, pd.DataFrame], steps: int
    ) -> pd.Series:
        """Recursive predict for multistep forecast using covariates.

        Same as :py:meth:`~_recursive_predict` but uses covariates.
        The regressor has been trained with covariates too in the
        fit call, otherwise error will be raised.


        Parameters
        ----------
        steps
            Number of predictions.
        future_covariates
            Exogenous variables.


        Returns
        -------
            Time series with the predictions indexed by
            time, starting from :py:attr:`~last_date`

        """

        self.__check_covariates_for_predict(future_covariates, steps)

        covariates_cols = (
            list(future_covariates.columns)
            if isinstance(future_covariates, pd.DataFrame)
            else [future_covariates.name]
        )
        # We need save observation for bootstraped methods
        observations_and_predictions = pd.DataFrame(
            np.nan,
            columns=[
                PREDICTIONS_COL_NAME,
                *self.lags_columns,
                *covariates_cols,
            ],
            index=pd.date_range(
                self.last_date, periods=steps + 1, freq=self.freq
            )[1:],
        )
        # Last window
        last_window = self.last_window

        for stp in range(steps):
            # obs is the observation needed to predict
            obs = np.array([*last_window, *future_covariates.iloc[stp]])
            # pred is a float with the prediction of the next step
            pred = self.regressor.predict(obs.reshape(1, -1))[0]
            # observations_and_predictions we save all predictions and observation
            observations_and_predictions.iloc[stp] = [pred, *obs]
            # We update value of last_window for the next iteration
            last_window = np.array([pred, *last_window[:-1]])
        # We use the same dtype as last_window + covariates
        # We save observation and predictions for boostrap interval predict
        self.observations_and_predictions = observations_and_predictions
        # We only return series of observations
        return observations_and_predictions[PREDICTIONS_COL_NAME]

    def predict(
        self,
        steps: int,
        future_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ):
        """Predicts a number of time ``steps`` forward.

        :py:meth:`~fit` method must be called first.
        If fitted with covariates, then ``future_covariates``
        must be provided in this method with an amount
        of observation equal to ``steps``. If fitted
        without covariates, then ``future_covariates``
        must be ``None``.


        Parameters
        ----------
        steps
            Number of predictions.
        future_covariates
            If provided, exogenous variables.


        Returns
        -------
            Time series with the predictions indexed by
            time, starting from :py:attr:`~last_date`

        """

        if not self.fitted:
            raise RegressorNotFittedError

        if not isinstance(steps, int) or steps < 1:
            raise ValueError(
                "`steps` must be positive integer greater or equal than 1."
            )

        if self.trained_with_covariates and future_covariates is None:
            raise ValueError(
                "If regressor has been trained with `future_covariates`, \
                    you need to provide it for prediction too."
            )

        if future_covariates is not None:
            return self._recursive_predict_with_covariates(
                future_covariates, steps
            )
        return self._recursive_predict(steps)

    @staticmethod
    def bootstrap_resample(
        X_train: np.ndarray, y_train: np.ndarray, percentage: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bootstrap over ``X_train`` matrix.

        In the probabilistic forecast using the
        bootstrap method, the ``X_train`` matrix
        using to fit the regressor, is partitioned
        using a resample with replacement to produce
        a new train matrix and validation matrix
        used to train and evaluate the boostrap sample
        regressor.


        Parameters
        ----------
        X_train
            Train matrix to fit the regressor.
        y_train
            Train target to fit the regressor.
        percentage
            Percentage of the train subsample.

        Returns
        -------
            _description_

        """
        training_ids = resample(
            range(len(X_train)), n_samples=int(percentage * len(X_train))
        )
        validation_ids = np.setdiff1d(range(len(X_train)), training_ids)
        return (
            X_train[training_ids],
            y_train[training_ids],
            X_train[validation_ids],
            y_train[validation_ids],
        )

    def __set_default_status(self):
        self.fitted = False
        self.trained_with_covariates = False
        self.covariates_names = None
        self.last_window = None
        self.last_date = None
        self.observations_and_predictions = None

    def bootstrap_predict_arround_observation(
        self,
        observation,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_bootstraps: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        n_samples = X_train.shape[0]
        # If n_bootstraps is not provided, we set it as the square root of n_samples
        n_bootstraps = (
            np.sqrt(n_samples).astype(int)
            if n_bootstraps is None
            else n_bootstraps
        )
        # Compute the m_i's and the validation residuals
        bootstrap_preds, val_residuals = [], []
        for _ in range(n_bootstraps):
            # Random sampling with replacement for training
            (
                X_train_boot,
                y_train_boot,
                X_validation_boot,
                y_validation_boot,
            ) = self.bootstrap_resample(X_train, y_train)

            # We fit regressor only in bootstraped training set
            self.regressor.fit(X_train_boot, y_train_boot)
            # We save residual for boot iteration
            val_residuals.append(
                np.mean(  # We take values because regressor of sckit-learn returns np.ndarray
                    y_validation_boot
                    - self.regressor.predict(X_validation_boot)
                )
            )
            # We predict our observation with the model trained using the bootstraped sample
            bootstrap_preds.append(self.regressor.predict(observation))
        # In this point, we return centered bootstrap prediction for observation
        bootstrap_preds -= np.mean(bootstrap_preds)
        return np.array(bootstrap_preds), np.array(val_residuals)

    @staticmethod
    def no_information_error(y_train: pd.Series, y_pred: pd.Series) -> float:
        return (
            np.sum(np.subtract.outer(y_train, y_pred) ** 2) / len(y_train) ** 2
        )

    @staticmethod
    def relative_overfitting_rate(
        no_information_error, train_residuals, val_residuals
    ):
        generalisation = np.abs(val_residuals.mean() - train_residuals.mean())
        no_information_val = np.abs(no_information_error - train_residuals)
        return np.mean(generalisation / no_information_val)

    @staticmethod
    def build_residuals(
        relative_overfitting_rate: float,
        train_residuals: np.ndarray,
        val_residuals: np.ndarray,
    ) -> np.ndarray:

        # Take percentiles of the training- and validation residuals to enable
        # comparisons between them
        val_residuals = np.percentile(val_residuals, q=np.arange(100))
        train_residuals = np.percentile(train_residuals, q=np.arange(100))
        weight = 0.632 / (1 - 0.368 * relative_overfitting_rate)
        return (1 - weight) * train_residuals + weight * val_residuals

    @staticmethod
    def probabilistic_interval(
        bootstrap_preds: np.ndarray,
        residuals: np.ndarray,
        lower_quantile: float,
        upper_quantile: float,
    ) -> Tuple[float, float]:
        # Construct the C set and get the percentiles
        prob_interval = np.array(
            [
                pred_bot + res
                for pred_bot in bootstrap_preds
                for res in residuals
            ]
        )
        percentiles = np.percentile(
            prob_interval, q=[lower_quantile, upper_quantile]
        )
        return percentiles[0], percentiles[1]

    def _predict_interval(
        self,
        observation,
        X_train: np.ndarray,
        y_train: np.ndarray,
        y_pred: pd.Series,
        lower_quantile: float,
        upper_quantile: float,
        n_bootstraps: Optional[int],
    ):
        if not self.fitted:
            raise RegressorNotFittedError
        # Here bootstrap_pred is an array of length n_bootstrap
        # with a prediction for observation for each bootstrap sample
        # and val_residuals is the mean of the validation error
        (
            bootstrap_preds,
            val_residuals,
        ) = self.bootstrap_predict_arround_observation(
            X_train=X_train,
            y_train=y_train,
            observation=observation,
            n_bootstraps=n_bootstraps,
        )
        # residuals for training
        train_residuals = y_train - y_pred
        # We compute the relative overfitting rate
        relative_overfitting_rate = self.relative_overfitting_rate(
            self.no_information_error(y_train, y_pred),
            train_residuals,
            val_residuals,
        )
        # We build residuals based on validation bootstrap errors and training error
        residuals = self.build_residuals(
            relative_overfitting_rate=relative_overfitting_rate,
            train_residuals=train_residuals,
            val_residuals=val_residuals,
        )
        # We return lower  percentile and upper percentile
        return self.probabilistic_interval(
            bootstrap_preds=bootstrap_preds,
            residuals=residuals,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    def predict_interval(
        self,
        y: pd.Series,
        steps: int,
        future_covariates: Optional[Union[pd.Series, pd.DataFrame]],
        lower_quantile,
        upper_quantile,
        n_bootstraps: Optional[int] = None,
    ):
        if not self.fitted:
            raise RegressorNotFittedError

        if future_covariates is not None:
            if len(future_covariates) != len(y) + steps:
                raise ValueError(
                    "`future_covariates` must contain the past (as many observation as `y`)"
                    "and the future (as many observation as steps)"
                )
            future_covariates_train, future_covariates_predict = (
                future_covariates[y.index],
                future_covariates[len(y) :],
            )
        else:
            future_covariates_train, future_covariates_predict = None, None
        # We create again X_train, y_train for bootstraping
        X_train, y_train = self.create_train_set(
            y, future_covariates=future_covariates_train
        )
        # We treat it as numpy ndarray
        X_train, y_train = X_train.values, y_train.values
        # We calculate predictions on training set
        y_pred = self.regressor.predict(X_train)
        # We calculate predictions, we only need observations saved on observations_and_predictions
        _ = self.predict(
            steps=steps, future_covariates=future_covariates_predict
        )
        # We get observations
        probabilistic_interval = pd.DataFrame(
            columns=[
                f"lower_quantile_{lower_quantile}",
                f"upper_quantile_{upper_quantile}",
                PREDICTIONS_COL_NAME,
            ],
            dtype=float,
            index=self.observations_and_predictions.index,
        )
        probabilistic_interval[
            PREDICTIONS_COL_NAME
        ] = self.observations_and_predictions[PREDICTIONS_COL_NAME]

        probabilistic_interval[
            [
                f"lower_quantile_{lower_quantile}",
                f"upper_quantile_{upper_quantile}",
            ]
        ] = self.observations_and_predictions.iloc[:, 1:].apply(
            lambda row: self._predict_interval(
                row.values.reshape(1, -1),
                X_train,
                y_train,
                y_pred,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                n_bootstraps=n_bootstraps,
            ),
            result_type="expand",
            axis=1,
        )
        # After bootstrap prediction, we set defaults, because
        # the model was re trained with multiples bootstrap samplings
        # So we impose re-training in order to don't get confuse
        self.__set_default_status()
        return probabilistic_interval


class RegressorForecastWithPredictors(RegressorForecast):
    def __init__(
        self,
        custom_predictors: Union[Callable, List[Callable]],
        predictors_window_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if (
            not isinstance(predictors_window_size, int)
            or predictors_window_size < 1
            or predictors_window_size < self.lags
        ):
            raise ValueError(
                "`predictors_window_size` must be positive integer and greater than the number of lags"
            )

        self.custom_predictors = (
            [custom_predictors]
            if not isinstance(custom_predictors, list)
            else custom_predictors
        )

        if not all(callable(pred) for pred in self.custom_predictors):
            raise ValueError(
                "Custom predictors must be a callable that acts on windows of a time serie."
            )

        self.predictors_window_size = predictors_window_size
        self.custom_predictors_columns = [
            f"custom_predictor_{n}"
            for n in range(1, len(self.custom_predictors) + 1)
        ]

    @property
    def min_train_series_length(self):
        return self.predictors_window_size + 1

    def create_lags(self, y: pd.Series) -> pd.DataFrame:
        """Overwrite lags method to add custom predictors cols."""
        # We create lags, all validations on y are made in the method
        lags_features = super().create_lags(y)
        # We build custom predictors
        custom_predictors_features = pd.concat(
            [
                y.rolling(self.predictors_window_size)
                .apply(predictor)
                .shift(1)
                for predictor in self.custom_predictors
            ],
            axis=1,
        )
        # We set column names
        custom_predictors_features.columns = self.custom_predictors_columns
        # We remove nan values of the last observations
        custom_predictors_features = custom_predictors_features[
            -(len(y) - self.predictors_window_size) :
        ]
        # Predictor custom contains more nans because self.predictors_window_size >= self.lags,
        # so we have to cut lags_features
        if custom_predictors_features.isnull().any().any():
            raise CustomPredictorsWithNullValues

        return pd.concat(
            [
                lags_features.loc[custom_predictors_features.index],
                custom_predictors_features,
            ],
            axis=1,
        )

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Not implemented yet")
