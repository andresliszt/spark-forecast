# -*- coding: utf-8 -*-
"""Some aggregations and filters for timeseries on Spark."""

from typing import Dict, List
from typing import Union

import pyspark
import pyspark.pandas as ps
import numpy as np
import pandas as pd
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers.reconciliation import MinTReconciliator


from spark_forecast.models.multivariate_forecasting import PREDICTIONS_COL_NAME
from spark_forecast.helpers._pyspark.utils import check_if_columns_in_dataframe
from spark_forecast.helpers import logger

# TODO VALIDACION PREVIA/POST JOIN

CODE_COLUMN = "code"
"""This will be the name of the code id column common to all hierarchies."""


class HierarchicalForecast:
    def __init__(
        self,
        item_column: str,
        local_column: str,
        target_column: str,
        time_column: str,
        hierarchical_master: Union[
            pyspark.sql.dataframe.DataFrame, ps.DataFrame
        ],
        hierarchical_mapping: List[str],
    ):

        """Implementation of ``darts`` hierarchical forecast on Spark

        Implements hiearchical reconciliation forecast
        for each local given by ``local_column``. This class
        is made in the context of supermarket pricing, where
        each sku that is sold there is classified among multiples
        hiearchies.

        ``hierarchical_master`` is an Spark dataframe
        containing information about hierachies and in this context
        hiearchies must be *decreasing*. What does the last mean?
        The dictionary ``hierarchical_mapping``  **maps**
        each hiearchy into its parent, and each parent column
        must contain less items that its son, and there exist
        only and only one parent. **Important** if there
        are multiple parents for a hiearchy, then the
        first one found is used.

        ``item_column`` is the name of the most *down* level
        in the hierarchy, for example, single sku time series.
        ``local_column`` as mentioned, is the name of the column
        of local (In ``darts`` context is the ``total``).
        ``time_column`` is the time column as always and
        ``target_column`` is the name of the column that we
        want to forecast.

        Example
        -------

        For example suppose we have a master in the form

        >>> hierarchical_master
        H1	H2	item_id
        H1_1	H2_1	A
        H1_1	H2_1	B
        H1_1	H2_2	C
        H1_1	H2_2	D
        H1_2	H2_3	E
        H1_2	H2_3	F
        H1_2	H2_4	G
        H1_2	H2_4	H

        We have three levels of hiearchies ``H1``, ``H2``
        and the last one ``item_id``. Note that they are
        strictly decreasing in terms of items per
        hiearchy. In this case ``H1`` is the most top
        level, then ``H2`` and finally ``item_id``.
        The mapping for this example must be

        >>> hierarchical_mapping
        {"H1": "total", "H2": "H1", "item_id": "H2"}

        The most top level must be mapped always to ``total``,
        that is an special word for ``darts``. How does this
        class build the mapping passed to ``darts``?.

        Given ``hierarchical_mapping`` it looks for each
        pair son/parent  the pair values in its columns
        after removing duplicates (as we mentioned, we assume
        that there exists only and only one parent always),
        for exmaple, ``H2_1`` should be mapped to ``H1_1``
        and ``H2_2`` to ``H1_1`` as well.


        See Also
        --------
        https://unit8co.github.io/darts/examples/16-hierarchical-reconciliation.html


        """

        logger.info(
            "``HierarchicalForecast`` inplements hiearchical forecast from darts."
            " This will be applied on Spark Data Frame containing time series for"
            " each item found in the ``item_column``. Is responsability of who use"
            " this class to ensure that all items in ``item_column`` starts and finishes"
            " in the same date, otherwise not controlled errors will be raised or unexpected"
            " behaivour from ``darts`` will occur. We recomend the usage of"
            " ``spark_forecast.helpers.time_series`` module to treat with dates."
            " ``hiearchical_master`` contains information for each item"
            " of ``item_column`` and we assume that is equals for each local"
            " found in ``local_column`` (i.e the master doesn't depend on the local)"
            " and is user responsability to ensure that this master will match"
            " the items found in the Spark Data Frame with time series information because"
            " a join operation will be performed and some skus may be lost if the master"
            " isn't complete.",
            item_column=item_column,
            local_column=local_column,
            time_column=time_column,
        )

        self.hierarchical_columns = list(hierarchical_mapping.keys())
        # We check if exists in dataframe
        check_if_columns_in_dataframe(
            hierarchical_master, self.hierarchical_columns
        )
        # We verify if mapping is valid
        for son_column, father_column in hierarchical_mapping.items():
            if father_column == "total":
                continue
            if (
                not hierarchical_master[son_column].nunique()
                >= hierarchical_master[father_column].nunique()
            ):
                raise ValueError(
                    "Each son column found in ``hierarchical_mapping`` must contain more"
                    "items than its parent."
                )

        # As always the time column
        self.time_column = time_column
        # The target column that we want to apply reconciliation
        self.target_column = target_column
        # Local column: 1 reconciliation forecast for each local
        self.local_column = local_column
        # The most down level (example skus)
        self.item_column = item_column
        # We only keep useful information from master
        self.hierarchical_master = self.__to_pandas_api(
            hierarchical_master[self.hierarchical_columns]
        )
        # We need this prefix to build pivoted table with aggregate information
        for hiearchical_column in self.hierarchical_columns:
            if hiearchical_column == self.item_column:
                # We don't need to add suffix in the item_column
                continue
            self.hierarchical_master[
                hiearchical_column
            ] += f"|{hiearchical_column}"

        self.hierarchical_mapping = hierarchical_mapping
        self._single_parent_mapping = None

    @property
    def single_parent_mapping(self) -> Dict[str, str]:
        """Mapping used internally for ``darts``"""
        if self._single_parent_mapping is None:
            mapping = {}
            for son_column, father_column in self.hierarchical_mapping.items():
                if father_column == "total":
                    mapping_table = self.hierarchical_master[
                        son_column
                    ].drop_duplicates()
                    _mapping = {
                        value: ["total"] for value in mapping_table.to_numpy()
                    }
                else:
                    mapping_table = self.hierarchical_master[
                        [son_column, father_column]
                    ].drop_duplicates(subset=son_column)
                    _mapping = {
                        getattr(row, son_column): [getattr(row, father_column)]
                        for _, row in mapping_table.iterrows()
                    }
                mapping = {**mapping, **_mapping}

            self._single_parent_mapping = mapping
            return self._single_parent_mapping
        return self._single_parent_mapping

    def __to_pandas_api(
        self, data: Union[pyspark.sql.dataframe.DataFrame, ps.DataFrame]
    ) -> Union[ps.DataFrame, List[ps.DataFrame]]:

        """We work in this class with pandas API"""

        if isinstance(data, ps.DataFrame):
            return data
        if isinstance(data, pyspark.sql.dataframe.DataFrame):
            try:
                # The newest version
                return data.pandas_api()
            except AttributeError:
                # Older versions
                return data.to_pandas_on_spark()

        if isinstance(data, list):
            return [self.__to_pandas_api(_data) for _data in data]

        raise TypeError(
            "Only supported for `pyspark.sql.dataframe.DataFrame` or `pyspark.pandas.DataFrame`"
        )

    def _hierarchical_aggregation(
        self,
        data: ps.DataFrame,
    ) -> List[ps.DataFrame]:

        """Builds sum aggregation per each hiearchy

        ``data`` contains all columns from
        :py:attr:`~hierarchical_columns` and we apply
        sum aggregation in order to build time series
        for each hiearchy.

        Returns
        -------
            A List of pyspark pandas Data Frame
            with time series for each hiearchy.
        """

        total = data.groupby(
            [self.local_column, self.time_column], as_index=False
        )[self.target_column].sum()

        # This will be a constant column needed to concat
        total[CODE_COLUMN] = "total"

        total = total.to_spark()

        return [
            total,
            *[
                data.groupby(
                    [self.local_column, hierchical_col, self.time_column],
                    as_index=False,
                )[self.target_column]
                .sum()
                .rename(columns={hierchical_col: CODE_COLUMN})
                .to_spark()
                if hierchical_col != self.item_column
                else data.rename(columns={hierchical_col: CODE_COLUMN})[
                    [
                        self.local_column,
                        self.time_column,
                        self.target_column,
                        CODE_COLUMN,
                    ]
                ].to_spark()
                for hierchical_col in self.hierarchical_columns
            ],
        ]

    def hierarchical_aggregation(
        self,
        data: ps.DataFrame,
    ) -> List[ps.DataFrame]:
        """Builds multiples hierchical aggregation (sum)"""

        data = self.__to_pandas_api(data)

        # We check suitable columns
        check_if_columns_in_dataframe(
            data,
            [
                self.item_column,
                self.local_column,
                self.time_column,
                self.target_column,
            ],
        )
        data_joined = data.merge(
            self.hierarchical_master,
            left_on=self.item_column,
            right_on=self.item_column,
            how="inner",
        )

        return self._hierarchical_aggregation(
            data=data_joined,
        )

    def __validate_input(self, data_inputs: List[ps.DataFrame]):

        if CODE_COLUMN not in data_inputs[0]:
            # If total data is at first position we create the constant column
            data_inputs[0][CODE_COLUMN] = "total"

        for data in data_inputs:
            data = self.__to_pandas_api(data)
            if PREDICTIONS_COL_NAME in data.columns:
                # If default prediction found, we replace it for target column
                # we need it in order to do a cancat
                data.rename(
                    columns={PREDICTIONS_COL_NAME: self.target_column},
                    inplace=True,
                )

            check_if_columns_in_dataframe(
                data,
                columns=[
                    self.target_column,
                    self.time_column,
                    self.local_column,
                    CODE_COLUMN,
                ],
                extra_msg="Each data pyspark pandas DataFrame given to do hiearchical"
                f" forecast must contain time column ``{self.time_column}``, "
                f" target column ``{self.target_column}`` and ``code`` column,"
                " where the last let us identify uniquely each time series comming.",
            )

    @staticmethod
    def _melt_hiearchical_forecast(
        reconciliated_forecasts,
        time_column,
        item_column,
        local,
    ):
        def get_hiearchy_type(code):
            # Inner hierchies are separated with |
            splited_code = code.split("|")
            try:
                return splited_code[1], splited_code[0]
            except IndexError:
                if splited_code[0] == "total":
                    return "total", splited_code[0]
                return item_column, splited_code[0]

        def get_hiearchy(col):
            return pd.Series([*get_hiearchy_type(col)])

        reconciliated_forecasts = reconciliated_forecasts.melt(
            id_vars=time_column,
            var_name="code_type",
            value_name=PREDICTIONS_COL_NAME,
        )
        reconciliated_forecasts[
            ["code_type", CODE_COLUMN]
        ] = reconciliated_forecasts["code_type"].apply(get_hiearchy)
        # Define constant column to identify local
        reconciliated_forecasts["local"] = local
        # We return the time column as ds, this is need for static Spark Schema
        return reconciliated_forecasts.rename(columns={time_column: "ds"})

    @staticmethod
    def reconcilation_forecast_grouped_data(
        data_local: pd.DataFrame,
        train_to,
        method: str,
        target_column: str,
        local_column: str,
        item_column: str,
        time_column: str,
        mapping: Dict[str, str],
    ) -> ps.DataFrame[
        "ds" : np.datetime64,
        "code_type":str,
        PREDICTIONS_COL_NAME:float,
        CODE_COLUMN:str,
        "local":str,
    ]:
        """Performs reconciliation forecast for a given local

        This function will be applied as a pandas UDF on
        a groupby operation. At this point ``data_local``
        is a pivoted table with a column for each different
        hiearchy code and item_id for a given local. It
        performs darts reconciliation forecast.


        """
        # We get the local code
        local = getattr(data_local.iloc[0], local_column)
        # We need datetime index
        data_local[time_column] = pd.to_datetime(data_local[time_column])
        # We pivot our table
        data_local = data_local.pivot_table(
            columns=CODE_COLUMN, values=target_column, index=time_column
        ).sort_index()
        # We infer the frequency, needed for darts
        data_local.index.freq = pd.infer_freq(data_local.index)
        # We eliminate nan cols. That is because some locals may
        # not contain a particular item_id
        data_local = data_local.dropna(axis=1)
        # We split into train/reconciliate
        train = data_local.loc[:train_to]
        # Darts has a bug if the dataframe contains named column index
        train.columns.name = None
        # We split the reconciliation transform data
        to_reconciliate = data_local[data_local.index > train.index[-1]]
        to_reconciliate.columns.name = None
        # We only get the items from this grouped data
        mapping = {
            col: mapping[col] for col in data_local.columns if col != "total"
        }
        # We transform into darts TimeSeries and we embbed the hierarchy
        train = TimeSeries.from_dataframe(train).with_hierarchy(mapping)

        to_reconciliate = TimeSeries.from_dataframe(
            to_reconciliate
        ).with_hierarchy(mapping)
        # We instantiate the darts reconciliator
        reconciliator = MinTReconciliator(method=method)
        # We fit with our pivoted
        reconciliator.fit(train)
        # Finally we transform
        reconciliated = (
            reconciliator.transform(to_reconciliate)
            .pd_dataframe()
            .reset_index()
        )

        return HierarchicalForecast._melt_hiearchical_forecast(
            reconciliated_forecasts=reconciliated,
            time_column=time_column,
            item_column=item_column,
            local=local,
        )

    def reconciliation_forecast(
        self,
        aggregate_data: Union[pyspark.sql.dataframe.DataFrame, ps.DataFrame],
        hierarchical_forecasts: List[
            Union[pyspark.sql.dataframe.DataFrame, ps.DataFrame]
        ] = 1,
        method: str = "ols",
    ):

        # We need it as pandas api
        aggregate_data = self.__to_pandas_api(aggregate_data)
        hierarchical_forecasts = self.__to_pandas_api(hierarchical_forecasts)

        # We validate
        self.__validate_input(aggregate_data)
        self.__validate_input(hierarchical_forecasts)

        # Our training set for reconciliation forecast
        train: List[ps.DataFrame] = ps.concat(aggregate_data, axis=0)
        # The forecasts to be reconciliated
        to_reconciliate: List[ps.DataFrame] = ps.concat(
            hierarchical_forecasts, axis=0
        )
        # This is the last date observed in train
        train_to = train[self.time_column].max()
        # Validation on min/max date of training/reconciliation set
        if train_to >= to_reconciliate[self.time_column].min():
            raise ValueError(
                f"Last date found in the training data is {train_to}."
                f" First date found in the forecast is {to_reconciliate[self.time_column].min()}"
                " which is not allowed because must be greater than the found in training."
            )
        # We build a big pivoted table in order to use it on groupby operation
        data_train_and_to_reconciliate = ps.concat(
            [train, to_reconciliate], axis=0
        )

        return (
            data_train_and_to_reconciliate.groupby(self.local_column)
            .apply(
                HierarchicalForecast.reconcilation_forecast_grouped_data,
                method=method,
                train_to=train_to,
                local_column=self.local_column,
                target_column=self.target_column,
                item_column=self.item_column,
                time_column=self.time_column,
                mapping=self.single_parent_mapping,
            )
            .to_spark()
        )
