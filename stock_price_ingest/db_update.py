import requests
import json
import pandas as pd
from datetime import datetime, timedelta, date
import yfinance as yf
from tqdm import tqdm
import re
from bs4 import BeautifulSoup

from config import FTSE350_ADDR, NASDAQ100_ADDR, SNP500_ADDR, DAX_ADDR, \
    CAC40_ADDR
from database.models import engine
from database import TickerCl, DailyPriceCl
from utils.string import to_snake_case


def clean_ticker(s: str, sub_dot: bool = True):
    # Remove . or \n from end
    s = re.sub(r"[\.\n]$", "", s)
    if sub_dot:
        # Sub all others for -
        s = re.sub(r"\.", "-", s)
    return s


def wiki_constituents(
        url, 
        ticker_idx, 
        company_idx,
        ticker_suffix: str = "",
        sub_dot: bool = True
        ):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")
    tickers = soup.find("table", {"id": "constituents"}) \
        .find("tbody") \
        .find_all("tr")
    tickers = [
        {
            "ticker": t.find_all("td")[ticker_idx].text,
            "company": t.find_all("td")[company_idx].text
        } for t in tickers if len(t.find_all("td")) > 0
    ]
    tickers_df = pd.DataFrame(tickers)
    tickers_df["ticker"] = [
        f"{clean_ticker(t, sub_dot)}{f'.{ticker_suffix}' if ticker_suffix != '' else ''}"
        for t in tickers_df["ticker"]]
    return tickers_df


if __name__ == "__main__":
    # Variables
    st_date = datetime(1980, 1, 1)
    # Market closes
    d = datetime.strptime(str(datetime.today())[:10], "%Y-%m-%d")
    market_close = {
        "FTSE350": datetime(d.year, d.month, d.day, 16, 30),
        "NASDAQ": datetime(d.year, d.month, d.day, 21, 0),
        "S&P500": datetime(d.year, d.month, d.day, 21, 0),
        "CAC40": datetime(d.year, d.month, d.day, 16, 30),
        "DAX": datetime(d.year, d.month, d.day, 16, 30),
    }
    en_date = {
        k:(v + timedelta(days=1)).date() if v < datetime.now()
        else datetime.now().date()
        for k, v in market_close.items()
    }

    content_df = pd.DataFrame()

    # FTSE 350
    url = FTSE350_ADDR
    payload = {
        "path": "ftse-constituents",
        "parameters": "indexname%3Dftse-350%26tab%3Dheatmap%26tabId%3Ddcd47cbd-346e-4bd0-bf77-039301c7d329",
        "components": [{"componentId": "block_content%3A72d8cb8c-5ef6-41a9-9bb9-49db0a064214", "parameters": None}]
    }
    headers = {
        "Content-Type": "application/json"
    }
    req = requests.post(url, data=json.dumps(payload), headers=headers)

    content = req.json()[0].get("content", [])
    if len(content) == 0:
        raise Exception("No content returned from request")
    content = content[0].get("value", {}).get("content", [])
    tickers_df = pd.DataFrame(content)[["tidm", "description"]] \
        .rename(columns={"tidm": "ticker", "description": "company"})
    # Update ticker to be .L
    tickers_df["ticker"] = [
        f"{clean_ticker(t)}.L" for t in tickers_df["ticker"]]
    tickers_df["market"] = "FTSE350"
    content_df = pd.concat([content_df, tickers_df])
    

    # NASDAQ100
    url = NASDAQ100_ADDR
    tickers_df = wiki_constituents(url, 1, 0)
    tickers_df["market"] = "NASDAQ"
    # Append to data
    content_df = pd.concat([content_df, tickers_df])

    # S&P 500
    url = SNP500_ADDR
    tickers_df = wiki_constituents(url, 0, 1)
    tickers_df["market"] = "S&P500"
    # Append to data
    content_df = pd.concat([content_df, tickers_df])

    # DAX
    url = DAX_ADDR
    tickers_df = wiki_constituents(url, 3, 1, sub_dot=False)
    tickers_df["market"] = "DAX"
    # Append to data
    content_df = pd.concat([content_df, tickers_df])

    # CAC 40
    url = CAC40_ADDR
    tickers_df = wiki_constituents(url, 3, 0, sub_dot=False)
    tickers_df["market"] = "CAC40"
    # Append to data
    content_df = pd.concat([content_df, tickers_df])

    ### ADD TICKERS TO DB ###
    # Ticker
    # Format data
    tickers = content_df[["ticker", "company", "market"]]
    # Remove any duplicate tickers (happend for NASDAQ AND S&P500)
    tickers = tickers.drop_duplicates(subset=["ticker", "market"])
    # Add last seen
    tickers["last_seen_date"] = date.today()
    # Upsert
    tickers = TickerCl().upsert_df(tickers, id_col=["ticker", "market"])

    # Get all tickers from db
    ticker_dates = pd.read_sql("""
        SELECT 
            t.ticker,
            t.market,
            MAX(dp.date) AS last_date
        FROM ticker AS t
        LEFT JOIN daily_price AS dp
            ON t.id = dp.ticker_id
        GROUP BY t.ticker, t.market
        """, engine)

    # Get new tickers
    ticker_dates = pd.merge(
        tickers,
        ticker_dates,
        on=["ticker", "market"],
        how="left"
    )
    # New tickers
    df_filter = ticker_dates["last_date"].isna()
    new_tickers = ticker_dates.loc[df_filter, :]
    new_tickers["last_date"] = st_date - timedelta(days=1)
    # Existing tickers
    existing_tickers = ticker_dates.loc[~df_filter, :]
    # Join together
    ticker_dates = pd.concat([new_tickers, existing_tickers], axis=0)
    ticker_dates["last_date"] = pd.to_datetime(ticker_dates.last_date)

    # Group tickers and dates
    date_groups = ticker_dates.groupby(["last_date", "market"], as_index=False)

    # Work with date groups
    for (ld, m), g in tqdm(date_groups, total=len(date_groups), desc="Date groups"):
        print(f"Processing {g.shape[0]} tickers for {m} ({ld.date()} < date < {en_date[m]})")
        ticker_list = g.ticker.unique().tolist()
        data = yf.download(
            ticker_list,
            start=ld.date(),
            end=en_date[m],
            interval='1d',
            group_by='Ticker',
            progress=False,
            auto_adjust=False,
            prepost=False,
            threads=True,
            proxy=None
        )
        if len(ticker_list) == 1:
            data["ticker"] = ticker_list[0]
            data = data.loc[~data["Close"].isnull()] \
                .drop_duplicates()
        else:
            # Reformat df

            def _meld_df(df, t):
                df["ticker"] = t
                df = df.loc[~df["Close"].isnull()] \
                    .drop_duplicates()
                return df
            data = pd.concat([_meld_df(data[t], t)
                            for t in data.columns.get_level_values(0).unique()])
        # Rename cols
        data = data.reset_index()
        cols_map = {k: to_snake_case(k) for k in data.columns}
        data = data.rename(columns=cols_map)
        # Find outliers - where proportion compared ot previous is too big or small
        # Must be a one off move (otherwise could be a share split)
        data_groups = data.groupby("ticker")
        for _, dg in data_groups:
            dg["close_prop"] = dg.close / dg.close.shift(1)
            dg_outl = ((dg.close_prop > 10) | (dg.close_prop < 0.1)) \
                & ((dg.close_prop.shift(-1) > 10) | (dg.close_prop.shift(-1) < 0.1))
            # Skip if none found
            if dg_outl.sum() == 0: 
                continue
            # Amend outlier columns
            for col in ["high", "low", "open", "close"]:
                dg.loc[dg_outl, col] = dg.loc[dg_outl, col] / dg.loc[dg_outl, "close_prop"]
            data.loc[dg.index] = dg
        # Calc fields
        data["change"] = data["close"] - data["open"]
        # Join on id
        data = pd.merge(
            data,
            g[["ticker", "id"]].rename(columns={"id": "ticker_id"}),
            on="ticker"
        )
        # Add to the database in groups
        batch_size = 10000
        for i in range(0, data.shape[0], batch_size):
            DailyPriceCl().upsert_df(
                data.iloc[i:i+batch_size], id_col=["ticker_id", "date"])
