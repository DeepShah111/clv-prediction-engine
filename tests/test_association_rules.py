import pandas as pd
from src.association_rules import _run_cooccurrence_fallback, _get_segment_customer_ids


def test_fallback_finds_the_tea_cup_pair():
    # 100 baskets: TEA and CUP appear together in 30, TEA alone in 10, CUP alone in 5, others in 55
    rows = ([{"TEA": True, "CUP": True, "X": False}] * 30 +
            [{"TEA": True, "CUP": False, "X": False}] * 10 +
            [{"TEA": False, "CUP": True, "X": False}] * 5 +
            [{"TEA": False, "CUP": False, "X": True}] * 55)
    basket = pd.DataFrame(rows).astype(bool)
    rules = _run_cooccurrence_fallback(basket)
    pairs = {(next(iter(a)), next(iter(c))) for a, c in zip(rules["antecedents"], rules["consequents"])}
    assert ("TEA", "CUP") in pairs and ("CUP", "TEA") in pairs
    assert (rules["lift"] > 1).all()


def test_segment_ids_accept_customer_id_underscore_column():
    seg = pd.DataFrame({"Customer_ID": ["100", "200", "300"],
                        "Segment_Name": ["High-Value Customers", "High-Value Customers", "Regular Customers"]})
    assert _get_segment_customer_ids(seg, "High-Value Customers") == {100, 200}
    assert _get_segment_customer_ids(seg, "All") is None