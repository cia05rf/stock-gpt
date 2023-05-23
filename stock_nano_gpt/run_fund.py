"""
STEPS:
----
1. Load data
2. Preprocess data
3. Load model
4. Use model to run fund
"""
# TODO: RECONFIGURE THIS TO FIT PROCESS DATA
import numpy as np
import torch
from sqlalchemy import create_engine
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os.path

from config import KEY, PERIOD, TARGET_PERIOD, NORM_DATA, STORE_PATH, \
    BATCH_SIZE, DB_PATH
from model import *  # This is done in order to run the model
from process_data import process_data
from fund.fund import Fund

# Build fp
fields = [KEY, str(PERIOD), str(TARGET_PERIOD)]
if NORM_DATA:
    fields.append("norm")
MODEL_FP = "_".join(fields)+".pt"
model_fp = os.path.join(STORE_PATH, MODEL_FP)
FUND_FP = "_".join(fields)+".csv"
fund_fp = os.path.join(STORE_PATH, FUND_FP)

# Load the model
model = torch.load(model_fp)
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
    a.close
FROM daily_price AS a
INNER JOIN ticker AS b
ON a.ticker_id = b.id 
WHERE date >= '2019-01-01' --ASSUMED AFTER TRAIN PERIOD
"""
daily_prices = pd.read_sql(sql, engine)

X, y, X_orig, y_orig, indexes = \
    process_data(daily_prices, "close", target_period=0)


def _resize(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])


X = _resize(X)
y = _resize(y)
X_orig = _resize(X_orig)
y_orig = _resize(y_orig)
indexes = _resize(indexes)

# Get the actuals
X_final = X[:, -1].reshape(X.shape[0])
X_orig_final = X_orig[:, -1].reshape(X_orig.shape[0])

batches = torch.tensor(X, dtype=torch.float32, device="cuda")

# Run the model to get the 5th day prediction
preds5 = torch.tensor([], dtype=torch.float32, device="cpu")
preds1 = torch.tensor([], dtype=torch.float32, device="cpu")
limit = None
if limit:
    for item in [X, X_final, X_orig, X_orig_final, indexes]:
        item = item[-limit:]
data_loader = torch.utils.data.DataLoader(
    batches, batch_size=BATCH_SIZE, shuffle=False)
for b in tqdm(data_loader, total=len(data_loader), desc=f"Making preds in batches of {BATCH_SIZE}"):
    new_preds = torch.tensor(model.generate(b, 5)[:, -1], device="cpu")
    preds5 = torch.concat([preds5, new_preds])
    new_preds = torch.tensor(model.generate(b, 1)[:, -1], device="cpu")
    preds1 = torch.concat([preds1, new_preds])

# Put into a dataframe
preds1_np = preds5.detach().cpu().numpy()
preds5_np = preds1.detach().cpu().numpy()
model_df = pd.DataFrame({
    "pred5": preds5_np,
    "pred1": preds1_np,
    "current": X_final,
    "close": X_orig_final,
}).join(pd.DataFrame(
    indexes, columns=["ticker", "date"]
))

# Adjust close price to £ rather than pence
model_df["close"] = model_df["close"] / 100

# Loop over as fund
fund = Fund(10000)
fund._verbose = False
n_max = 10  # Number of items to buy at a time
model_df = model_df.sort_values(["date", "pred5"], ascending=[True, False])
model_df_grouped = model_df.groupby("date")
for date, df in tqdm(model_df_grouped, total=len(model_df_grouped), desc="Run fund"):
    # Buy the best
    if fund.available > 0:
        val = fund.available / n_max
        df_buy = df[
            (df["close"] != 0)
            & (df["pred5"] > df["current"])
            & (df["pred1"] > df["current"])
        ].head(n_max)
        for i, row in df_buy.iterrows():
            try:
                # Skip if already invested
                if fund.cur_holdings.get(row["ticker"], {}).get("share_vol", 0) > 0:
                    continue
                fund.buy(row["ticker"], date, row["close"], 0.01, val)
            except Exception as e:
                print(e)
    # Sell any items where pred5 and pred1 are less than close
    df_sell = df[
        (df["pred5"] <= df["current"])
        & (df["pred1"] <= df["current"])
        & (df["ticker"].isin([k for k,v in fund.cur_holdings.items() if v.get("share_vol", 0) > 0]))
    ]
    for i, row in df_sell.iterrows():
        try:
            fund.sell(row["ticker"], date, row["close"], 0.01)
        except Exception as e:
            print(e)

# Print summary
fund.summary()

# Ouput fund to csv
fund.ledger.to_csv(fund_fp, index=False)

# # Plot some of the output
# fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 9))
# sns.lineplot(x=range(preds.shape[0]), y=preds[:, 0], ax=ax1, label="preds")
# sns.lineplot(x=range(actuals.shape[0]), y=actuals[:, 0], ax=ax1, label="actuals")
# sns.lineplot(x=range(preds.shape[0]), y=preds[:, 1], ax=ax2, label="preds")
# sns.lineplot(x=range(actuals.shape[0]), y=actuals[:, 1], ax=ax2, label="actuals")
# fig.show()
