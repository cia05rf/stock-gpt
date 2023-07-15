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

from config import KEY, PERIOD, TARGET_PERIOD, STORE_PATH, \
    BATCH_SIZE, DB_PATH, SEQ_TYPE, TRANS_TYPE, NORM_SIZE
from gpt_model import *
from process_data import process_data


def compound(t: torch.Tensor):
    t *= 0.01  # convert from xx% to 0.xx
    t += 1  # make relative to 1.0
    return t


def nothing(t: torch.Tensor):
    return t


PRED_TYPE = "norm_close"
# per_gain == multiply generations to compound
# norm_close == normalised close price, adjust by finding range and
#   adding proportionally to min
GENERATE_CALLBACK = nothing

# Build fp
fields = [KEY, SEQ_TYPE, f"period{PERIOD}", f"pred{TARGET_PERIOD}"]
if TRANS_TYPE == "norm":
    fields += ["norm", f"normsize{NORM_SIZE}"]
elif TRANS_TYPE:
    fields += [TRANS_TYPE]
MODEL_FP = "_".join(fields)+".pt"
model_fp = os.path.join(STORE_PATH, MODEL_FP)

if __name__ == "__main__":
    # Load the model
    model = torch.load(model_fp).to("cuda")
    """
    PROCESS:
    1. Fetch data by day and process into sequences
    2. Run the model 5 iters to get prediction for 5 days
    3. Join back on to get actuals rather than normalised
    4. Identify and rank buys and sells
    """
    # Calc start date for fetch
    start_date_str = (datetime.today() - timedelta(days=200)) \
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
        a.close,
        a.change
    FROM daily_price AS a
    INNER JOIN ticker AS b
    ON a.ticker_id = b.id
    WHERE date >= '{start_date_str}'
    """
    daily_prices = pd.read_sql(sql, engine)
    daily_prices["change_per"] = daily_prices.change / daily_prices.close
    daily_prices = daily_prices.sort_values(["ticker_id", "date"])
    daily_prices["dod_change_per"] = 100 * (daily_prices.close - daily_prices.close.shift(1)) \
        / daily_prices.close.shift(1)

    # Get the last n dates in here
    dates = daily_prices.date.drop_duplicates()
    dates = dates.sort_values()
    dates = dates.iloc[-PERIOD:].to_frame()
    # dates = dates.iloc[-PERIOD-5:-5].to_frame() # TEMP - use to look back 5 days

    # Filter to only the last n days
    daily_prices = pd.merge(
        daily_prices,
        dates,
        on="date"
    )

    X, X_orig, indexes, close = \
        process_data(
            daily_prices,
            KEY,
            norm_data=TRANS_TYPE == "norm"
        )

    # Get the actuals
    X_final = X[:, -1]
    close_final = close[:, -1]
    indexes_final = indexes[:, -1]
    # Get the range for each item (eg between 0 and 1)
    lim = NORM_SIZE if NORM_SIZE < PERIOD else None
    close_range = (close[:, :NORM_SIZE].max(
        axis=1) - close[:, :NORM_SIZE].min(axis=1)).flatten()
    
    batches = torch.tensor(X, dtype=torch.float32, device="cuda")

    # Run the model to get the 5th day prediction
    preds5 = torch.tensor([], dtype=torch.float32, device="cpu")
    preds1 = torch.tensor([], dtype=torch.float32, device="cpu")
    limit = None
    if limit:
        for item in [X, X_final, X_orig, close_final, indexes]:
            item = item[-limit:]
    data_loader = torch.utils.data.DataLoader(
        batches, batch_size=BATCH_SIZE, shuffle=False)
    for b in tqdm(data_loader, total=len(data_loader), desc=f"Making preds in batches of {BATCH_SIZE}"):
        new_preds = torch.tensor(model.generate(b, 1)[:, -1], device="cpu")
        preds1 = torch.concat([preds1, new_preds])
        if PRED_TYPE == "per_gain":
            # Compound last 5
            new_preds = torch.tensor(model.generate(b, 5), device="cpu")
            new_preds = GENERATE_CALLBACK(new_preds)
            new_preds = new_preds[:, -5:].prod(axis=1)
        else:
            # Otherwise just take the last one
            new_preds = torch.tensor(model.generate(b, 5), device="cpu")
            new_preds = GENERATE_CALLBACK(new_preds)[:, -1]
        preds5 = torch.concat([preds5, new_preds])

    # Put into a dataframe
    preds1_np = preds1.detach().cpu().numpy()
    preds5_np = preds5.detach().cpu().numpy()
    model_df = pd.DataFrame({
        "pred5": preds5_np,
        "pred1": preds1_np,
        "current": X_final,
        "close": close_final,
        "range": close_range,
    }).join(pd.DataFrame(
        indexes_final, columns=["ticker_id", "date"]
    ))
    # Adjust close price to £ rather than pence
    model_df["close"] = model_df["close"] / 100
    model_df["range"] = model_df["range"] / 100
    # Calculate the change
    if PRED_TYPE == "per_gain":
        # make 0 the minimum
        model_df.loc[model_df.pred5 < 0, "pred5"] = 0
        model_df.loc[model_df.pred1 < 0, "pred1"] = 0
        model_df["pred5_close"] = model_df.close * model_df.pred5
        model_df["pred1_close"] = model_df.close * model_df.pred1
    elif PRED_TYPE == "norm_close":
        model_df["pred5_close"] = model_df.close \
            + ((model_df.pred5 - model_df.current) * model_df.range)
        model_df["pred1_close"] = model_df.close \
            + ((model_df.pred1 - model_df.current) * model_df.range)
    else:
        model_df["pred5_close"] = model_df.pred5
        model_df["pred1_close"] = model_df.pred1
    model_df["per_change5"] = (model_df.pred5_close -
                               model_df.close) / model_df.close

    # Mark buys and sells
    buy_filter = (
        (model_df["close"] != 0)
        & (model_df["pred1_close"] > model_df["close"])
        # & (model_df["pred5_close"] > model_df["close"])
    )
    model_df.loc[buy_filter, "signal"] = "buy"
    sell_filter = (
        (model_df["pred1_close"] <= model_df["close"])
        # & (model_df["pred5_close"] <= model_df["close"])
    )
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
    model_df["change"] = model_df["pred1_close"] - model_df["close"]
    model_df["change_per"] = model_df["change"] / model_df["close"]
    model_df = model_df[["ticker", "close", "pred1_close", "pred5_close",
                         "signal", "change", "change_per", "date"]]
    # Display buys
    buys = model_df[model_df.signal == "buy"].sort_values(
        "change_per", ascending=False)
    buys.to_csv(os.path.join(
        "./out", f"buys{datetime.today().strftime('%Y%m%d')}.csv"), index=False)
    print(buys.head(10))
    # Display sells
    sells = model_df[model_df.signal == "sell"].sort_values(
        "change_per", ascending=True)
    sells.to_csv(os.path.join(
        "./out", f"sells{datetime.today().strftime('%Y%m%d')}.csv"), index=False)
    print(sells.head(10))
