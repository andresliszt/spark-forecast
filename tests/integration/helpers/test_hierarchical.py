import pytest
import pyspark.pandas as ps


from spark_forecast.helpers._pyspark.hierarchical import HierarchicalForecast

TEST_DATA = [
    (
        {
            "H1": [
                "H1_1",
                "H1_1",
                "H1_1",
                "H1_1",
                "H1_2",
                "H1_2",
                "H1_2",
                "H1_2",
            ],
            "H2": [
                "H2_1",
                "H2_1",
                "H2_2",
                "H2_2",
                "H2_3",
                "H2_3",
                "H2_4",
                "H2_4",
            ],
            "item_id": ["A", "B", "C", "D", "E", "F", "G", "H"],
        },
        {"item_id": "H2", "H2": "H1", "H1": "total"},
        {
            "A|item_id": ["H2_1|H2"],
            "B|item_id": ["H2_1|H2"],
            "C|item_id": ["H2_2|H2"],
            "D|item_id": ["H2_2|H2"],
            "E|item_id": ["H2_3|H2"],
            "F|item_id": ["H2_3|H2"],
            "G|item_id": ["H2_4|H2"],
            "H|item_id": ["H2_4|H2"],
            "H2_1|H2": ["H1_1|H1"],
            "H2_2|H2": ["H1_1|H1"],
            "H2_3|H2": ["H1_2|H1"],
            "H2_4|H2": ["H1_2|H1"],
            "H1_1|H1": ["total"],
            "H1_2|H1": ["total"],
        },
    )
]


@pytest.mark.parametrize(
    "master, hiearchy, expected_darts_mapping_dict", TEST_DATA
)
def test_mapping(master, hiearchy, expected_darts_mapping_dict):
    hier = HierarchicalForecast(
        item_column="NotUsedHere",
        local_column="NotUsedHere",
        target_column="NotUsedHere",
        time_column="NotUsedHere",
        hierarchical_master=ps.DataFrame(master),
        hierarchical_mapping=hiearchy,
    )
    assert hier.single_parent_mapping == expected_darts_mapping_dict
