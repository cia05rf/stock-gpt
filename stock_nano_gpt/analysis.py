import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Import the data
ledger = pd.read_csv("./out/fund.csv")

# Close out trades
buys = ledger[ledger["trade_type"] == "BUY"] \
    [["ticker", "trade_date", "ledger_value", "trade_type"]]
sells = ledger[ledger["trade_type"] == "SELL"] \
    [["ticker", "trade_date", "ledger_value", "trade_type"]]
l = pd.merge(buys, sells, on=["ticker"], suffixes=["_buy", "_sell"])
l = l[l.trade_date_buy < l.trade_date_sell]
m = l.groupby(["ticker", "trade_date_buy"]).agg({"trade_date_sell": "min"}) \
    .reset_index()
l = pd.merge(l, m, on=["ticker", "trade_date_buy", "trade_date_sell"])
l["net"] = l.ledger_value_buy + l.ledger_value_sell
l["yield"] = l.net / -l.ledger_value_buy
l = l.replace([np.inf, -np.inf], np.nan)
# Remove anything with a yield > 1
l = l[l["yield"] < 1]
l["bin"] = pd.cut(l["yield"], 20)

# Plot yield buckets
fig, ax = plt.subplots(figsize=(16,9))
plt_df = l.groupby(["bin"]).agg({"bin": "count"}) \
    .rename(columns={"bin": "count"})
plt_df.index.name = "bin"
plt_df = plt_df.reset_index()
sns.barplot(x="bin", y="count", data=plt_df, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

# Agg and cumsum
l_date = l.groupby("trade_date_sell").agg({"net": "sum", "ledger_value_buy": "sum"}) \
    .reset_index()
l_date["yield"] = l_date.net / -l.ledger_value_buy
l_date["cumsum"] = l_date.net.cumsum()


# Plot the available post trade by date
fig, ax = plt.subplots(figsize=(16,9))
plt_df = l_date[l_date.trade_date_sell.between("2020-05-01", "2020-07-31")]
sns.lineplot(x="trade_date_sell", y="cumsum", data=plt_df, ax=ax)
ax.set_xticks(ax.get_xticks()[::5])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
_ = 1