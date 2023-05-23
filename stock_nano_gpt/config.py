"""Config file for running scripts"""
import json
import os
import re

CONFIG = []

with open("./config.json", "r") as f:
    CONFIG = json.loads(f.read())

# Overwrite with local config if present
if os.path.isfile("./local.config.json"):
    with open("local.config.json", "r") as f:
        local_config = json.loads(f.read())
        for k, v in local_config.items():
            # Replace the first part APPSETTING_
            if k in CONFIG:
                CONFIG[k] = v

# Overwrite with env vars if duplicated
for k, v in os.environ.items():
    # Replace the first part APPSETTING_
    k = re.sub(r"^APPSETTING_", "", k)
    if k in CONFIG:
        CONFIG[k] = v

# Assign to variables
# Scraping
FTSE350_ADDR = CONFIG.get("web_scrape", {}).get("ftse350", "https://api.londonstockexchange.com/api/v1/components/refresh")
# Files
STORE_PATH = CONFIG.get("files", {}).get("store_path", "./data")
# Database
DB_PATH = os.path.join(STORE_PATH, CONFIG.get("db", {}).get("filename", "prices.db"))
DB_NAME = CONFIG.get("db", {}).get("name", "prices")
DB_UPDATE_PRICES = CONFIG.get("db_update", {}).get("prices", "full")
DB_UPDATE_SIGNALS = CONFIG.get("db_update", {}).get("signals", "full")
# Misc
PUBLIC_HOLS = CONFIG.get("public_holidays", [])
# Sequence
PERIOD = CONFIG.get("sequence", {}).get("period", 30)
TARGET_PERIOD = CONFIG.get("sequence", {}).get("target_period", 1)
NORM_DATA = CONFIG.get("sequence", {}).get("normalise", True)
KEY = CONFIG.get("sequence", {}).get("key", "close")
# Model
BATCH_SIZE = CONFIG.get("model", {}).get("batch_size", 32)