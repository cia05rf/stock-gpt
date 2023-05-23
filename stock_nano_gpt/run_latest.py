"""
STEPS:
----
1. Load data
2. Preprocess data
3. Load model
4. Use model to run fund
"""

import torch
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import os.path

from config import KEY, PERIOD, TARGET_PERIOD, NORM_DATA, STORE_PATH, \
    BATCH_SIZE, DB_PATH
from model import *
from process_data import process_data

# Build fp
fields = [KEY, str(PERIOD), str(TARGET_PERIOD)]
if NORM_DATA:
    fields.append("norm")
MODEL_FP = "_".join(fields)+".pt"
model_fp = os.path.join(STORE_PATH, MODEL_FP)

if __name__ == "__main__":
    # Load the model
    model = torch.load(model_fp)
    """
    PROCESS:
    1. Fetch data by day and process into sequences
    2. Run the model 5 iters to get prediction for 5 days
    3. Join back on to get actuals rather than normalised
    4. Identify and rank buys and sells
    """
    # Calc start date for fetch
    start_date_str = (datetime.today() - timedelta(days=100)) \
        .strftime('%Y-%m-%d')
    # Fetch data
    # Start the engine and Session
    engine = create_engine(
        f'sqlite:///{DB_PATH}'
    )
    sql = f"""
    SELECT 
        a.ticker_id,
        b.ticker,
        a.date,
        a.close
    FROM daily_price AS a
    INNER JOIN ticker AS b
    ON a.ticker_id = b.id 
    WHERE date >= '{start_date_str}'
    """
    daily_prices = pd.read_sql(sql, engine)

    # Get the last n dates in here
    dates = daily_prices.date.unique()
    dates.sort()
    dates = dates[-PERIOD:]

    # Filter to only the last n days
    daily_prices = daily_prices[daily_prices.date.isin(dates)]

    X, y, X_orig, y_orig, indexes = \
        process_data(daily_prices, target_period=0)
    def _resize(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    X = _resize(X)
    y = _resize(y)
    X_orig = _resize(X_orig)
    y_orig = _resize(y_orig)
    # Keep only the last index
    indexes = indexes[:, -1, :]

    # Get the actuals
    X_final = X[:, -1].reshape(X.shape[0])
    X_orig_final = X_orig[:, -1].reshape(X_orig.shape[0])

    # Get the scaling factor
    # To get to the original value, we need to do:
    # (X * X_orig_size) + X_orig_min
    X_orig_min = np.min(X_orig, axis=-1)
    X_orig_size = np.max(X_orig, axis=-1) - X_orig_min

    X = torch.tensor(X, dtype=torch.float32, device="cuda")

    # Run the model to get the 5th day prediction
    preds5 = torch.tensor([], dtype=torch.float32, device="cpu")
    preds1 = torch.tensor([], dtype=torch.float32, device="cpu")
    limit = None
    if limit:
        for item in [X, X_final, X_orig, X_orig_final, X_orig_min, X_orig_size,
                    indexes]:
            item = item[-limit:]
    data_loader = torch.utils.data.DataLoader(
        X, batch_size=BATCH_SIZE, shuffle=False)
    for b in tqdm(data_loader, total=len(data_loader), desc=f"Making preds in batches of {BATCH_SIZE}"):
        new_preds = torch.tensor(model.generate(b, 5)[:, -1], device="cpu")
        preds5 = torch.concat([preds5, new_preds])
        new_preds = torch.tensor(model.generate(b, 1)[:, -1], device="cpu")
        preds1 = torch.concat([preds1, new_preds])

    # Put into a dataframe
    preds1_np = preds1.detach().cpu().numpy()
    preds5_np = preds5.detach().cpu().numpy()
    model_df = pd.DataFrame({
        "pred5": preds5_np,
        "pred1": preds1_np,
        "target5": (preds5_np * X_orig_size) + X_orig_min,
        "target1": (preds1_np * X_orig_size) + X_orig_min,
        "current": X_final,
        "close": X_orig_final
    }).join(pd.DataFrame(
        indexes, columns=["ticker_id", "date"]
    ))

    # Mark buys and sells
    buy_filter = (model_df["close"] != 0) \
        & (model_df["pred5"] > model_df["current"]) \
        & (model_df["pred1"] > model_df["current"])
    model_df.loc[buy_filter, "signal"] = "buy"
    sell_filter = (model_df["pred5"] <= model_df["current"]) \
            & (model_df["pred1"] <= model_df["current"])
    model_df.loc[sell_filter, "signal"] = "sell"
    model_df.loc[model_df.signal.isna(), "signal"] = "hold"

    # Add ticker info
    sql = f"""
    SELECT 
        id AS ticker_id,
        ticker,
        company
    FROM ticker
    """
    tickers = pd.read_sql(sql, engine)

    model_df = model_df.merge(tickers, on="ticker_id")
    model_df = model_df[["ticker", "close", "target1", "target5", "signal"]]
    model_df["change"] = model_df["target5"] - model_df["close"]
    model_df["change_per"] = model_df["change"] / model_df["close"]
    # Display buys
    buys = model_df[model_df.signal == "buy"].sort_values("change_per", ascending=False)
    buys.to_csv(os.path.join("./out", f"buys{datetime.today().strftime('%Y%m%d')}.csv"), index=False)
    print(buys.head(10))
    # Display sells
    sells = model_df[model_df.signal == "sell"].sort_values("change_per", ascending=True)
    sells.to_csv(os.path.join("./out", f"sells{datetime.today().strftime('%Y%m%d')}.csv"), index=False)
    print(sells.head(10))