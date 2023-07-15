"""
STEPS:
----
1. Load data
2. Preprocess data
3. Load model
4. Use model to run fund
"""
import numpy as np
import torch
from sqlalchemy import create_engine
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os.path

from config import KEY, PERIOD, TARGET_PERIOD, STORE_PATH, \
    BATCH_SIZE, DB_PATH, SEQ_TYPE, TRANS_TYPE, NORM_SIZE
from gpt_model import *  # This is done in order to run the model
from process_data import process_data
from fund.fund import Fund


# Reverse transformation methods


def rev_trans_compound(
        preds: np.ndarray,
        close: np.ndarray,
        target_period: int,
        *args, **kwargs
) -> np.ndarray:
    # Convert percentage to decimal
    preds *= 0.01  # convert from xx% to 0.xx
    preds += 1  # make relative to 1.0
    # Limit to > 0
    preds = preds.clip(min=0.0)
    # Compound gains/losses
    comp = np.prod(preds[:, -target_period:], axis=1)
    comp = close * comp
    return comp


def rev_trans_norm(
        preds: np.ndarray,
        close: np.ndarray,
        orig: np.ndarray,
        cl_range: np.ndarray,
        target_period: int,
        *args, **kwargs
) -> np.ndarray:
    current = orig[:, -1]
    mins = close - (current * cl_range)
    out = mins + (cl_range * preds[:, -target_period])
    return out


def rev_trans_norm_to_first(
        preds: np.ndarray,
        close: np.ndarray,
        target_period: int,
        *args, **kwargs
) -> np.ndarray:
    preds = (preds.T * close[:, 0]).T
    return preds[:, -target_period]


def rev_trans_log(preds: np.ndarray,
                  target_period: int, *args, **kwargs) -> np.ndarray:
    return np.exp(preds[:, -target_period])


def rev_trans_nothing(preds: np.ndarray,
                      target_period: int, *args, **kwargs) -> np.ndarray:
    return preds[:, -target_period]


# Set the function
if TRANS_TYPE == "norm":
    rev_trans_func = rev_trans_norm
elif TRANS_TYPE == "norm_to_first":
    rev_trans_func = rev_trans_norm_to_first
elif TRANS_TYPE == "log":
    rev_trans_func = rev_trans_log
elif TRANS_TYPE == "compound":
    rev_trans_func = rev_trans_compound
else:
    rev_trans_func = rev_trans_nothing


# Build fp
fields = [KEY, SEQ_TYPE, f"period{PERIOD}", f"pred{TARGET_PERIOD}"]
if TRANS_TYPE == "norm":
    fields += ["norm", f"normsize{NORM_SIZE}"]
elif TRANS_TYPE:
    fields += [TRANS_TYPE]
MODEL_FP = "_".join(fields)+".pt"
model_fp = os.path.join(STORE_PATH, MODEL_FP)
FUND_FP = "fund_" + "_".join(fields) + ".csv"
fund_fp = os.path.join(STORE_PATH, FUND_FP)

if __name__ == "__main__":
    """
    PROCESS:
    1. Fetch data by day and process
    2. Run the model 5 iters to get prediction for 5 days
    3. Join back on to get actuals rather than normalised
    3. Use for fund:
        - Buy top prediction
        - If own and prediction is negative or 0, sell
    4. Repeat over all days for one year from end of dataset
    """
    # Calc start date for
    # Fetch data
    # Start the engine and Session
    engine = create_engine(
        f'sqlite:///{DB_PATH}'
    )
    sql = """
    SELECT 
        a.ticker_id,
        b.ticker,
        a.date,
        a.close,
        a.change
    FROM daily_price AS a
    INNER JOIN ticker AS b
    ON a.ticker_id = b.id 
    WHERE date >= '2022-01-01' --ASSUMED AFTER TRAIN PERIOD
    """
    daily_prices = pd.read_sql(sql, engine)
    daily_prices["change_per"] = daily_prices.change / daily_prices.close
    daily_prices = daily_prices.sort_values(["ticker_id", "date"])
    daily_prices["dod_change_per"] = 100 * (daily_prices.close - daily_prices.close.shift(1)) \
        / daily_prices.close.shift(1)

    X, X_orig, indexes, close = \
        process_data(
            daily_prices,
            KEY,
        )

    # Get the actuals
    X_final = X_orig[:, -1]
    close_final = close[:, -1]
    indexes_final = indexes[:, -1]
    # Get the range for each item (eg between 0 and 1)
    lim = NORM_SIZE if NORM_SIZE < PERIOD else None
    close_range = (close[:, :NORM_SIZE].max(
        axis=1) - close[:, :NORM_SIZE].min(axis=1)).flatten()

    batches = torch.tensor(X, dtype=torch.float32, device="cuda")

    # Run the model to get the 5th day prediction
    preds = torch.tensor([], dtype=torch.float32, device="cpu")
    limit = None
    if limit:
        for item in [X, X_orig, X_final, close_final, indexes]:
            item = item[-limit:]
    data_loader = torch.utils.data.DataLoader(
        batches, batch_size=BATCH_SIZE, shuffle=False)
    # Load the model
    model = torch.load(model_fp).to("cuda")
    for b in tqdm(data_loader, total=len(data_loader), desc=f"Making preds in batches of {BATCH_SIZE}"):
        new_preds = torch.tensor(model.generate(b, 5), device="cpu")
        preds = torch.concat([preds, new_preds])

    # Put into a dataframe
    model_df = pd.DataFrame({
        "close": close_final,
        "pred1_close": rev_trans_func(
            preds=preds.detach().cpu().numpy(),
            close=close_final,
            orig=X,
            cl_range=close_range,
            target_period=1,
            norm_size=NORM_SIZE
        ),
        "pred5_close": rev_trans_func(
            preds=preds.detach().cpu().numpy(),
            close=close_final,
            orig=X,
            cl_range=close_range,
            target_period=5,
            norm_size=NORM_SIZE
        ),
        "range": close_range,
        "current": X_final,
        "pred1": preds[:, -5].detach().cpu().numpy(),
        "pred5": preds[:, -1].detach().cpu().numpy(),
    }).join(pd.DataFrame(
        indexes_final, columns=["ticker", "date"]
    ))

    # Adjust close price to £ rather than pence
    for col in ["close", "pred1_close", "pred5_close", "range"]:
        model_df[col] = model_df[col] / 100
    model_df["per_change1"] = (model_df.pred1_close -
                               model_df.close) / model_df.close
    model_df["per_change5"] = (model_df.pred5_close -
                               model_df.close) / model_df.close

    # Add actuals
    model_df["actual1"] = None
    model_df["actual5"] = None
    groups = model_df.sort_values(["ticker", "date"]).groupby("ticker")
    for _, g in tqdm(groups, total=len(groups), desc="Adding actuals"):
        g["actual1"] = g.close.shift(-TARGET_PERIOD)
        g["actual5"] = g.close.shift(-TARGET_PERIOD*5)
        model_df.loc[g.index] = g
    # Reorder columns
    model_df = model_df[[
        "ticker", "date", "close", "actual1", "pred1_close", "per_change1",
        "actual5", "pred5_close", "per_change5"]]

    ## PLOT ##
    plt_df = model_df[model_df.ticker == 1]
    plt_df = plt_df.reset_index(drop=True)

    # Plot some of the output
    fig, (ax1) = plt.subplots(nrows=1, figsize=(16, 9))
    plt.xticks(rotation=30)
    sns.lineplot(x=plt_df.date, y=plt_df.close, ax=ax1, label="close")
    # # 1 step
    # sns.lineplot(x=plt_df.date, y=plt_df.actual1, ax=ax1, label="close1")
    # sns.lineplot(x=plt_df.date, y=plt_df.pred1_close,
    #              ax=ax1, label="pred1_close")
    # 5 step
    sns.lineplot(x=plt_df.date, y=plt_df.actual5, ax=ax1, label="close5")
    sns.lineplot(x=plt_df.date, y=plt_df.pred5_close,
                 ax=ax1, label="pred5_close")
    ax1.xaxis.set_major_locator(plt.MaxNLocator(20))
    fig.show()

    # RUN FUND
    fund = Fund(10000)
    fund._verbose = False
    n_max = 10  # Number of items to buy at a time
    model_df = model_df.sort_values(["date"], ascending=[True])
    model_df_grouped = model_df.groupby("date")
    for date, df in tqdm(model_df_grouped, total=len(model_df_grouped), desc="Run fund"):
        # Buy the best
        if fund.available > 0:
            val = fund.available / n_max
            df_buy = df[
                (df["close"] != 0)
                & (df["pred1_close"] > df["close"])
                & (df["pred5_close"] > df["close"])
            ]
            df_buy = df_buy.sort_values(["per_change5"], ascending=False) \
                .head(n_max)
            for i, row in df_buy.iterrows():
                try:
                    # Skip if already invested
                    if fund.cur_holdings.get(row["ticker"], {}).get("share_vol", 0) > 0:
                        continue
                    fund.buy(row["ticker"], date, row["close"], 0.01,
                             val, signal_prob=row["per_change5"])
                except Exception as e:
                    print(e)
        # Sell any items where pred5 and pred1 are less than close
        df_sell = df[
            (df["ticker"].isin(
                [k for k, v in fund.cur_holdings.items() if v.get("share_vol", 0) > 0]))
            & (df["pred1_close"] <= df["close"])
            & (df["pred5_close"] <= df["close"])
        ]
        for i, row in df_sell.iterrows():
            try:
                fund.sell(row["ticker"], date, row["close"],
                          0.01, signal_prob=row["per_change5"])
            except Exception as e:
                print(e)

    # Print summary
    fund.summary()

    # Ouput fund to csv
    fund.ledger.to_csv(fund_fp, index=False)
