"""Small feature helpers with no heavy dependencies (testable without lifetimes)."""
import pandas as pd


def _compute_return_rate(raw_df, customer_index, split_date):
    """
    Share of each customer's invoices (before split_date) that were returns.
    Customer IDs are converted exactly like clean_data does (float -> Int64 -> str),
    so they match the text IDs in customer_index.
    """
    ids   = (pd.to_numeric(raw_df['Customer ID'], errors='coerce')
             .astype('Int64').astype(str))
    dates = pd.to_datetime(raw_df['InvoiceDate'])
    keep  = (dates < split_date) & ids.isin(customer_index)

    sub = raw_df.loc[keep].copy()
    sub['Customer ID'] = ids[keep]

    total_orders  = sub.groupby('Customer ID')['Invoice'].nunique()
    return_orders = (sub[sub['Quantity'] < 0]
                     .groupby('Customer ID')['Invoice'].nunique())

    rate = (return_orders / total_orders).fillna(0).clip(0, 1)
    return rate.reindex(customer_index, fill_value=0.0)