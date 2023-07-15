"""Takes the data from the database and saves it to an npy file."""

import pandas as pd
from typing import List
import numpy as np
from tqdm import tqdm
import os
from datetime import timedelta

from config import STORE_PATH, PERIOD, TARGET_PERIOD, KEY, \
    SEQ_TYPE, VAL_DAYS
from database.models import engine
from utils.data import norm_np

def single_long_sequence(
        g: pd.DataFrame,
        seq_len: int = PERIOD + TARGET_PERIOD,
        key: str = KEY
        ) -> np.ndarray:
    """Builds a sequence of data for a single ticker.
    Args:
        g (pd.DataFrame): The data for a single ticker. 
    Returns:
        np.ndarray: The data for the ticker.
    """
    # Sort by date
    g = g.sort_values('date')
    # Set the index
    g = g.set_index('date')
    # If too short then pad with zeros to right
    g_data = g[key].to_numpy()
    if g.shape[0] < seq_len:
        g_data = np.pad(
            g_data,
            (0, seq_len - g.shape[0]),
            'constant',
            constant_values=np.nan
            )
    return g_data

def multi_short_sequences(
        g: pd.DataFrame,
        seq_len: int = PERIOD + TARGET_PERIOD,
        key: str = KEY
        ) -> np.ndarray:
    """Builds a sequence of data for a single ticker.
    Args:
        g (pd.DataFrame): The data for a single ticker.
    Returns:
        np.ndarray: The data for the ticker.
    """
    # Sort by date
    g = g.sort_values('date')
    # Set the index
    g = g.set_index('date')
    # Model as key over period
    new_data = []
    # If too short then pad with zeros to right
    g_data = g[key].to_numpy()
    if g.shape[0] < seq_len:
        g_data = np.pad(
            g_data,
            (0, seq_len - g.shape[0]),
            'constant',
            constant_values=np.nan
            )
    new_data = np.array([
        g_data[st:st + seq_len]
        for st in range(0, g_data.shape[0] - seq_len + 1)
    ])
    return new_data

def pad_to_max(sequences: List[np.ndarray]) -> List[np.ndarray]:
    max_len = max([d.shape[0] for d in sequences])
    for i, d in enumerate(sequences):
        sequences[i] = np.pad(
            d,
            (0, max_len - d.shape[0]),
            'constant',
            constant_values=np.nan
            )
    return sequences


if __name__ == "__main__":
    # Build fp
    fields = [KEY, SEQ_TYPE]
    out_fn = "_".join(fields)+".np"
    out_val_fn = "_".join(fields+["val"])+".np"
    fp = os.path.join(STORE_PATH, out_fn)
    val_fp = os.path.join(STORE_PATH, out_val_fn)


    ### READ DATA ###
    SQL = """
        SELECT * FROM daily_price
        WHERE date > '2001-01-01'
    """
    daily_prices_df = pd.read_sql(SQL, engine)
    daily_prices_df["date"] = pd.to_datetime(daily_prices_df.date)

    ### FORMAT TO GROUPS ###
    daily_prices_df["change_per"] = daily_prices_df.change / daily_prices_df.close
    daily_prices_df = daily_prices_df.sort_values(["ticker_id", "date"])
    daily_prices_df["dod_change_per"] = 100 * (daily_prices_df.close - daily_prices_df.close.shift(1)) \
        / daily_prices_df.close.shift(1)
    daily_prices_df = daily_prices_df[["ticker_id", "date", KEY]]
    # Group by ticker
    daily_prices_grouped = daily_prices_df.groupby('ticker_id')
    # Start file
    if os.path.exists(fp):
        os.remove(fp)
    data = []
    val_data = []
    for _, g in tqdm(daily_prices_grouped,
                    total=len(daily_prices_grouped),
                    desc="Building data"):
        # Select VAL_DAYS random days to leave out for testing
        # Must be at least VAL_DAYS*2 days from end to allow for end training
        if g.shape[0] < VAL_DAYS*2:
            g["val"] = False
        else:
            days = np.random.randint(VAL_DAYS*2, g.shape[0])
            val_st_date = g.date.max() - timedelta(days=days)
            val_en_date = val_st_date + timedelta(days=VAL_DAYS)
            g.loc[g.date.between(val_st_date, val_en_date), "val"] = True
            g["val"] = g.val.fillna(False)
        # Identify validation data
        v_data = g.loc[g.val].copy()
        # Overwrite in train set
        g.loc[g.val, KEY] = np.nan
        # Get sequences
        if SEQ_TYPE == "single_long":
            new_data = single_long_sequence(g)
            new_val_data = single_long_sequence(v_data)
        else:
            new_data = multi_short_sequences(g)
            new_val_data = multi_short_sequences(v_data)
        # Append to data
        data.append(new_data)
        val_data.append(new_val_data)

    # Bulk to right if single_long
    if SEQ_TYPE == "single_long":
        data = pad_to_max(data)
        val_data = pad_to_max(val_data)
        # Concatenate data
        data = np.array(data)
        val_data = np.array(val_data)
    else:
        # Concatenate data
        data = np.concatenate(data, axis=0)
        val_data = np.concatenate(val_data, axis=0)
    # Save data
    # Data saved as 180 cols of features and final col is target
    with open(fp, 'wb') as f:
        np.save(f, data)
        print("Saved data to", fp)
    with open(val_fp, 'wb') as f:
        np.save(f, val_data)
        print("Saved data to", val_fp)
