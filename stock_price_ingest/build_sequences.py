"""Takes the data from the database and saves it to an npy file."""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from datetime import timedelta

from config import STORE_PATH, PERIOD, TARGET_PERIOD, NORM_DATA, \
    KEY
from database.models import engine
from utils.data import norm_np

VAL_DAYS = 100

if __name__ == "__main__":
    # Build fp
    fields = [KEY, str(PERIOD), str(TARGET_PERIOD)]
    if NORM_DATA:
        fields.append("norm")
    out_fn = "_".join(fields)+".np"
    out_val_fn = "_".join(fields+["val"])+".np"
    fp = os.path.join(STORE_PATH, out_fn)
    val_fp = os.path.join(STORE_PATH, out_val_fn)


    ### READ DATA ###
    SQL = """SELECT * FROM daily_price"""
    daily_prices_df = pd.read_sql(SQL, engine)

    # Select 100 random days to leave out for testing
    # Must be at least 200 days from end to allow for end training
    daily_prices_df["date"] = pd.to_datetime(daily_prices_df.date)
    max_val_st_date = daily_prices_df.date.max() - timedelta(days=200)
    dates = daily_prices_df.date[daily_prices_df.date < max_val_st_date]
    val_st_date = dates.sample(1).max()
    val_en_date = val_st_date + timedelta(days=VAL_DAYS)
    val_data_df = daily_prices_df.loc[daily_prices_df.date.between(val_st_date, val_en_date)]
    val_data_df["val"] = True

    ### FORMAT TO GROUPS ###
    daily_prices_df["change_per"] = daily_prices_df.change / daily_prices_df.close
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
        # Skip if too small
        if g.shape[0] < PERIOD + TARGET_PERIOD:
            continue
        # Identify validation data
        val = g.join(val_data_df, rsuffix="_").val \
            .fillna(False)
        # Sort by date
        g = g.sort_values('date')
        # Set the index
        g = g.set_index('date')
        # Model as key over period
        new_data = []
        new_data = np.array([
            g.iloc[st:st + PERIOD + TARGET_PERIOD][KEY].to_numpy()
            for st in range(0, g.shape[0] - PERIOD - TARGET_PERIOD)
        ])
        new_data = np.array(new_data)
        # Append to data
        data.append(new_data[~val[:new_data.shape[0]]])
        val_data.append(new_data[val[:new_data.shape[0]]])
    # Concatenate data
    data = np.concatenate(data, axis=0)
    val_data = np.concatenate(val_data, axis=0)
    # Normalise over data axis=1
    if NORM_DATA:
        # foo[:, -TARGET_PERIOD] for the last columns
        data = norm_np(data, axis=1, holdout=TARGET_PERIOD)
        val_data = norm_np(val_data, axis=1, holdout=TARGET_PERIOD)
    # Save data
    # Data saved as 180 cols of features and final col is target
    with open(fp, 'wb') as f:
        np.save(f, data)
    with open(val_fp, 'wb') as f:
        np.save(f, val_data)
