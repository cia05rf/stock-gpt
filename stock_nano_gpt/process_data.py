"""Takes the data from the database and saves it to an hdf file."""
from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from config import PERIOD, KEY, VAL_LIMIT, NORM_SIZE, TRANS_TYPE


def ffill(arr: np.ndarray) -> np.ndarray:
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


def bfill(arr: np.ndarray) -> np.ndarray:
    arr = arr[:, ::-1]
    arr = np.where(arr == 0, np.nan, arr)

    def ffill(arr):
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:, None], idx]
        return out
    arr = ffill(arr)
    arr = arr[:, ::-1]
    return arr

### NORMA FUNCTIONS ###
# Transformation methods


def trans_norm(arr: np.ndarray, norm_size: int = NORM_SIZE) -> np.ndarray:
    # Back fill any leading 0s
    arr = bfill(arr)
    # Limit the normalisation group to block_size
    # This will allow for values >1 and <0 in the block
    lim = norm_size if arr.shape[1] > norm_size else None
    arr_lim = arr[:, :lim]
    maxs = (
        np.nanmax(arr_lim, axis=1) * np.ones_like(arr.T)
    ).T
    mins = (
        np.nanmin(arr_lim, axis=1) * np.ones_like(arr.T)
    ).T
    out = ((arr - mins) / (maxs - mins))
    # Avoid inf
    # (this will happen where it's the same number throughout)
    out = np.where(
        maxs != mins,
        out, 0
    )
    # Set a max and min limit
    if VAL_LIMIT is not None:
        out = np.where(out > VAL_LIMIT, VAL_LIMIT, out)
        out = np.where(out < -VAL_LIMIT, -VAL_LIMIT, out)
    return out


def trans_norm_to_first(arr: np.ndarray) -> np.ndarray:
    # Back fill any leading 0s
    arr = bfill(arr)
    arr2 = arr[:, 0]
    arr2 = (arr2 * np.ones_like(arr.T)).T
    arr = arr / arr2
    return arr


def trans_log(arr: np.ndarray) -> np.ndarray:
    return np.log(np.nan_to_num(arr, nan=0, neginf=0, posinf=0))


def trans_inf(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0, neginf=-1, posinf=1)


# Set the function
if TRANS_TYPE == "norm":
    trans_func = trans_norm
elif TRANS_TYPE == "norm_to_first":
    trans_func = trans_norm_to_first
elif TRANS_TYPE == "log":
    trans_func = trans_log
else:
    trans_func = trans_inf


def process_data(
        daily_prices: pd.DataFrame,
        key: str = KEY,
        period: int = PERIOD
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
        norm_data (bool, optional): _description_. Defaults to False.
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
    # Start file
    daily_prices_grouped = daily_prices.groupby('ticker_id', as_index=False)
    data = []
    close = []
    indexes = []
    for _, g in tqdm(daily_prices_grouped,
                     total=len(daily_prices_grouped),
                     desc="Building data"):
        # Skip if too small
        if g.shape[0] < period:
            continue
        # Sort by date
        g = g.sort_values('date')
        # Fill missing data
        g[key] = g[key].fillna(method='ffill')
        # Model as normalised close over period
        new_data = np.array([
            g[key].to_numpy()
        ])
        new_close = np.array([
            g["close"].values
        ])
        new_indexes = np.array([
            g[["ticker_id", "date"]].values
        ])
        # Append to data
        data.append(new_data)
        close.append(new_close)
        indexes.append(new_indexes)
    # Concatenate data
    data = np.concatenate(data, axis=0)
    close = np.concatenate(close, axis=0)
    indexes = np.concatenate(indexes, axis=0)
    # Build batches (1 batch == all tickers over a dates range)
    # batches = np.array([
    #     [
    #         data[j, i:i+period] for i in
    #         range(max(data.shape[1] - period, 1))
    #     ]
    #     for j in range(data.shape[0])
    # ])
    # close = np.array([
    #     [
    #         close[j, i:i+period] for i in
    #         range(max(close.shape[1] - period, 1))
    #     ]
    #     for j in range(close.shape[0])
    # ])
    # # Limit ranges of index
    # if indexes.shape[1] > period:
    #     indexes = indexes[:, period:]
    # Reshape

    def _reshape(arr: np.ndarray) -> np.ndarray:
        arr = np.array([
            [
                arr[j, i:i+period] for i in
                range(max(arr.shape[1] - period, 1))
            ]
            for j in range(arr.shape[0])
        ])
        return arr.reshape(np.prod(arr.shape[:2]), *arr.shape[2:])
    # Reshape
    batches = _reshape(data)
    indexes = _reshape(indexes)
    close = _reshape(close)
    # Limit other ranges
    X_orig = batches.copy()
    X = batches
    # Transform X
    X = trans_func(X)
    # Fillnas
    X = np.where(np.isnan(X), 0, X)
    X_orig = np.where(np.isnan(X_orig), 0, X_orig)
    close = np.where(np.isnan(close), 0, close)
    return X, X_orig, indexes, close
