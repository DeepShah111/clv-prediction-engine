import pandas as pd
from src.segment_utils import name_clusters


def test_highest_value_cluster_gets_the_top_name_whatever_its_number():
    # K-Means numbered the HIGH-value group 0 here (the opposite of your real run)
    names = name_clusters(pd.Series({0: 2342.0, 1: 181.0}))
    assert names[0] == "High-Value Customers"
    assert names[1] == "Regular Customers"


def test_three_clusters_are_ranked():
    names = name_clusters(pd.Series({0: 50.0, 1: 900.0, 2: 300.0}))
    assert names == {1: "High-Value Customers", 2: "Mid-Value Customers", 0: "Low-Value Customers"}