"""Takes the data from the database and saves it to an hdf file."""
from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from config import PERIOD, TARGET_PERIOD, NORM_DATA, KEY

def process_data(
        daily_prices: pd.DataFrame,
        key: str = KEY,
        period: int = PERIOD,
        target_period: int = TARGET_PERIOD,
        norm_data: bool = NORM_DATA
) -> List[np.ndarray]:
    """
    Takes in a dataframe of daily prices and returns a list of numpy arrays
    of shape (n, m, o) where:
        n is the number of sequences
        m is the number of tickers
        o is the length of the sequence

    Args:
        daily_prices (pd.DataFrame): _description_
        key (str): _description_
        period (int, optional): _description_. Defaults to PERIOD.
        target_period (int, optional): _description_. Defaults to TARGET_PERIOD.
        target (int, optional): _description_. Defaults to TARGET.
        norm_data (bool, optional): _description_. Defaults to NORM_DATA.
        round_dp (int, optional): _description_. Defaults to ROUND_DP.

    Returns:
        List[np.ndarray]: [batches to predict, actuals, index]
    """
    # Fill in date gaps
    dates = daily_prices[["date"]].drop_duplicates()
    dates["key"] = 1
    ticker_ids = daily_prices[["ticker_id"]].drop_duplicates()
    ticker_ids["key"] = 1
    base = pd.merge(ticker_ids, dates, on="key").drop(columns=["key"])
    daily_prices = pd.merge(daily_prices, base, on=[
                            "ticker_id", "date"], how="right")
    # Drop any duplicates
    daily_prices = daily_prices.drop_duplicates(["ticker_id", "date"])
    daily_prices[key] = daily_prices[key].fillna(0)
    # Start file
    daily_prices_grouped = daily_prices.groupby('ticker_id', as_index=False)
    data = []
    indexes = []
    for _, g in tqdm(daily_prices_grouped,
                  total=len(daily_prices_grouped),
                  desc="Building data"):
        # Skip if too small
        if g.shape[0] < period + target_period:
            continue
        # Sort by date
        g = g.sort_values('date')
        # Model as normalised close over period
        new_data = np.array([
            g[key].to_numpy()
        ])
        new_indexes = np.array([
            g[["ticker_id", "date"]].values
        ])
        # Append to data
        data.append(new_data)
        indexes.append(new_indexes)
    # Concatenate data
    data = np.concatenate(data, axis=0)
    indexes = np.concatenate(indexes, axis=0)
    # Build batches (1 batch == all tickers over a dates range)
    batches = np.array([
        [
            data[j, i:i+period+target_period] for i in
            range(data.shape[1] - period - target_period)
        ]
        for j in range(data.shape[0])
    ])
    # Normalise over data axis=1
    lim = -target_period if target_period > 0 else None
    y_orig = batches.copy()[:, :, lim:] if lim is not None \
        else batches.copy()[:, :, 0:0]
    X_orig = batches.copy()[:, :, :lim]
    X = batches
    if norm_data:
        v = X[:, :, :lim]
        min_ = np.expand_dims(np.min(v, axis=-1), axis=-1)
        max_ = np.expand_dims(np.max(v, axis=-1), axis=-1)
        X = ((X - min_) / (max_ - min_))
    X = X[:, :, :lim]
    y = X[:, :, lim:] if lim is not None \
        else X[:, :, 0:0]
    # Fillnas
    X = np.where(np.isnan(X), 0, X)
    y = np.where(np.isnan(y), 0, y)
    y_orig = np.where(np.isnan(y_orig), 0, y_orig)
    X_orig = np.where(np.isnan(X_orig), 0, X_orig)
    # Remove final indexes
    indexes = indexes[:, period:lim, :]
    return X, y, X_orig, y_orig, indexes
