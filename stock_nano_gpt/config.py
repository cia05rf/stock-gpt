"""Config file for running scripts"""
import json
import os
import re

CONFIG = []

with open("C:/Users/robert.franklin/Desktop/local_projects/random/stock-gpt/config.json", "r") as f:
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
# Files
STORE_PATH = CONFIG.get("files", {}).get("store_path", "./data")
# Database
DB_PATH = os.path.join(STORE_PATH, CONFIG.get("db", {}).get("filename", "prices.db"))
# Sequence
PERIOD = CONFIG.get("sequence", {}).get("period", 30)
TARGET_PERIOD = CONFIG.get("sequence", {}).get("target_period", 1)
NORM_SIZE = CONFIG.get("sequence", {}).get("norm_period", 30)
KEY = CONFIG.get("sequence", {}).get("key", "close")
SEQ_TYPE = CONFIG.get("sequence", {}).get("seq_type", "single_long")
TRANS_TYPE = CONFIG.get("sequence", {}).get("trans_type", None)
VAL_LIMIT = CONFIG.get("sequence", {}).get("value_limit", None)
# Model
BATCH_SIZE = CONFIG.get("model", {}).get("batch_size", 32)
MAX_ITERS = CONFIG.get("model", {}).get("max_iters", 10000)
LEARNING_RATE = CONFIG.get("model", {}).get("learning_rate", 6e-4)
DROPOUT = CONFIG.get("model", {}).get("dropout", 0.2)
N_LAYER = CONFIG.get("model", {}).get("n_layer", 12)
N_HEAD = CONFIG.get("model", {}).get("n_head", 16)
N_EMBD = CONFIG.get("model", {}).get("n_embd", 256)