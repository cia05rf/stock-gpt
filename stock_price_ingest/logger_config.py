"""Configuration file for the logging. Used in libs.logs"""
import yaml
from datetime import datetime

from config import CONFIG

# Add logger config if given
LOGGER_CONFIG = {}
with open("./stock_trading_ml_modelling/logger_config.yaml", "r") as stream:
    try:
        LOGGER_CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

level = CONFIG.get("logs", {}).get("log_level", "INFO")
datetime_str = datetime.strftime(datetime.now(), "%Y%m%d")

# Overwrite where neccessary
LOGGER_CONFIG["loggers"]["main"]["level"] = level
LOGGER_CONFIG["handlers"]["std_file"]["filename"] = f"./logs/{datetime_str}_stdout.log"
LOGGER_CONFIG["handlers"]["err_file"]["filename"] = f"./logs/{datetime_str}_errout.log"
