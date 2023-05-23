"""
Script for createing a new database from existing price data
held in a h5 file
"""
import os

from config import DB_PATH
from database.models import engine, Base

if __name__ == "__main__":
    #Delete the old files
    try:
        os.remove(DB_PATH)
        print(f'\nSUCCESSFULLY REMOVED {DB_PATH}')
    except Exception as e:
        print(f'\nERROR - REMOVING:{e}')
    # Build database
    Base.metadata.create_all(engine)