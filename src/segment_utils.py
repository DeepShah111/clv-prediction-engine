"""Profile-based segment naming (no heavy imports, easy to test)."""
import pandas as pd


def name_clusters(values_by_cluster):
    """
    Names clusters by RANKING their average value, so the names cannot be swapped by
    the arbitrary cluster numbering of K-Means.
    values_by_cluster: pandas Series, index = cluster number, values = average spend/CLV.
    Returns {cluster_number: segment_name}, the highest-value cluster first.
    """
    ranked = values_by_cluster.sort_values(ascending=False).index.tolist()
    k = len(ranked)
    if k == 2:
        labels = ["High-Value Customers", "Regular Customers"]
    elif k == 3:
        labels = ["High-Value Customers", "Mid-Value Customers", "Low-Value Customers"]
    else:
        labels = (["High-Value Customers"]
                  + [f"Mid-Value Tier {i}" for i in range(1, k - 1)]
                  + ["Low-Value Customers"])
    return {int(cluster): label for cluster, label in zip(ranked, labels)}