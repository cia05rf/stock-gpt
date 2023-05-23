import pandas as pd

from utils.log import logger

from modelling.data_builder import DataBuilder

folder = "new_cnn"
limit_id = None
years = 2
refresh = False

#Import predictions
data = DataBuilder(
    limit_id=limit_id,
    folder=folder
)
preds, labels = data.load_preds()

#Load data
data.limit_id = limit_id
data.get_price_data(weeks=52*years)
data.load_data(keys=["prices_id_test"])

#Rejoin onto dataframe
data.prices = data.prices.set_index("id")
prices = data.prices.loc[
    data.prices_id_test
]
prices["signal"] = preds

#Run fund
prices = prices.sort_values(["date","ticker_id"])