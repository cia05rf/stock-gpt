"""Create sub-classes for querying the database"""
from sqlalchemy import func
from typing import List
import logging
import pandas as pd

from database.models import Session as session
from database.models.prices import Ticker, DailyPrice, WeeklyPrice
from database.manage_data import _add_df, _update_df


class Cl:
    id_col = ["id"]

    def __init__(self):
        pass

    def _fetch_to_df(self, df: pd.DataFrame, id_col: List[str] = None) -> pd.DataFrame:
        """Function for converting a query to a dataframe"""
        if id_col is None:
            id_col = self.id_col
        records = session.query(self.dest_class)
        for ic in id_col:
            records = records.filter(getattr(self.dest_class, ic).in_(df[ic]))
        records = records.all()
        return pd.DataFrame([r.__dict__ for r in records])

    def add_df(self, df, id_col: List[str] = None, session=session) -> pd.DataFrame:
        """Function to add data to the database.

        args:
        ----
        df - pandas dataframe - the data to be added to the database
        session - sqla session:None - the db session object

        returns:
        ----
        None
        """
        df = df.drop_duplicates()
        _add_df(df, self.dest_class, session=session)
        return self._fetch_to_df(df, id_col)

    def update_df(self, df, id_col: List[str] = None, session=session) -> pd.DataFrame:
        """Function for updating records from a dataframe"""
        _update_df(df, self.dest_class, session=session)
        return self._fetch_to_df(df, id_col)

    def upsert_df(self, df, id_col: List[str] = None, session=session) -> pd.DataFrame:
        """Function for upserting records from a dataframe"""
        # Query table to get existing records
        existing = session.query(self.dest_class)
        for ic in id_col:
            existing = existing.filter(
                getattr(self.dest_class, ic).in_(df[ic]))
        existing = existing.all()
        if len(existing):
            existing_df = pd.DataFrame([e.__dict__ for e in existing])
            existing_df = pd.merge(
                df, existing_df,
                how="left", on=id_col, suffixes=("", "_y"),
                indicator=True
            )
            add_df = existing_df[existing_df["_merge"]
                                 == "left_only"][df.columns]
            update_df = existing_df[existing_df["_merge"] == "both"]
            # Update existing items
            self.update_df(update_df, session=session)
        else:
            add_df = df
        # Add new items
        self.add_df(add_df, session=session)
        return self._fetch_to_df(df, id_col)


class TickerCl(Cl):
    dest_class = Ticker

    def __init__(self):
        super().__init__()

    def add_df(self, df, session=session):
        """Function to add data to the database.

        args:
        ----
        df - pandas dataframe - the data to be added to the database
        session - sqla session:None - the db session object

        returns:
        ----
        None
        """
        if df.shape[0]:
            keep_cols = ['ticker', 'company', 'market']
            if 'last_seen_date' in df.columns:
                keep_cols.append('last_seen_date')
            df = df[keep_cols] \
                .drop_duplicates()
            _add_df(df, Ticker, session=session)

    def fetch(self,
              ticker_ids=[],
              from_date=None,
              to_date=None
              ):
        """Function to create a query to grab all tickers from the offers db.

        args:
        ----
        session - sqla session
        ticker_ids - list:[] - the ids of the records to be extracted
        from_date - dtatime:None - the min date for filtering records
        to_date - dtatime:None - the max date for filtering records

        returns:
        ----
        sqlalchemy query 
        """
        query = session.query(Ticker)
        if len(ticker_ids):
            query = query.filter(Ticker.id.in_(ticker_ids))
        if from_date:
            query = query.filter(Ticker.last_seen_date >= from_date)
        if to_date:
            query = query.filter(Ticker.last_seen_date <= to_date)
        return query

class DailyPriceCl(Cl):
    dest_class = DailyPrice

    def __init__(self):
        super().__init__()

    def add_df(self, df, session=session):
        """Function to add data to the database.

        args:
        ----
        df - pandas dataframe - the data to be added to the database
        session - sqla session:None - the db session object

        returns:
        ----
        None
        """
        if df.shape[0]:
            df = df[['date', 'open', 'high', 'low', 'close', 'change', 'volume', 'ticker_id']] \
                .drop_duplicates()
            _add_df(df, DailyPrice, session=session)

    def fetch(self,
              ticker_ids=[],
              from_date=None,
              to_date=None
              ):
        """Function to create a query to grab all sub-jobs from the offers db.
        Open sub-jobs are set to status_id 1.

        args:
        ----
        ticker_ids - list:[] - the ids of the records to be extracted

        returns:
        ----
        sqlalchemy query 
        """
        query = session.query(DailyPrice)
        if len(ticker_ids):
            query = query.filter(DailyPrice.ticker_id.in_(ticker_ids))
        if from_date:
            query = query.filter(DailyPrice.date >= from_date)
        if to_date:
            query = query.filter(DailyPrice.date <= to_date)
        return query

    def fetch_latest(self,
                     session,
                     ticker_ids=[],
                     from_date=None,
                     to_date=None
                     ):
        """Function to get that last entry for each item

        args:
        ----
        session - sqla session
        ticker_ids - list:[] - the ids of the records to be extracted
        from_date - dtatime:None - the min date for filtering records
        to_date - dtatime:None - the max date for filtering records

        returns:
        ----
        sqla query
        """
        # create the sub-query
        subq = session.query(
            DailyPrice.ticker_id,
            func.max(DailyPrice.date).label("max_date")
        )
        # filter for dates
        if from_date:
            subq = subq.filter(DailyPrice.date >= from_date)
        if to_date:
            subq = subq.filter(DailyPrice.date <= to_date)
        # order the results
        subq = subq.order_by(DailyPrice.ticker_id, DailyPrice.date.desc()) \
            .group_by(DailyPrice.ticker_id) \
            .subquery("t2")
        # build the main query
        query = session.query(Ticker, subq.c.max_date) \
            .outerjoin(
                subq,
                subq.c.ticker_id == Ticker.id
        )
        # filter on ticker ids wanted
        if len(ticker_ids):
            query = query.filter(Ticker.id.in_(ticker_ids))
        return query

    def remove(self,
               ids=[],
               ticker_ids=[],
               from_date=None,
               to_date=None,
               del_all=False
               ):
        """Function to delete records from the daily prices table.

        args:
        ----
        ids - list:[] - the ids of the records to be extracted
        ticker_ids - list:[] - the ticker ids of the records to be extracted
        from_date - dtatime:None - the min date for filtering records
        to_date - dtatime:None - the max date for filtering records
        del_all - bool:False - safety to prevet deleting the whole table

        returns:
        ----
        sqlalchemy query 
        """
        try:
            # Preform check to prevent del_all
            if not del_all and not len(ids) and not len(ticker_ids) and not from_date and not to_date:
                logging.warning(
                    "Delete not performed as no attributes given and del_all is False")
                return False
            query = session.query(DailyPrice)
            if len(ids):
                query = query.filter(DailyPrice.id.in_(ids))
            if len(ticker_ids):
                query = query.filter(DailyPrice.ticker_id.in_(ticker_ids))
            if from_date:
                query = query.filter(DailyPrice.date >= from_date)
            if to_date:
                query = query.filter(DailyPrice.date <= to_date)
            query.delete(synchronize_session=False)
            session.commit()
            return True
        except:
            return False


class WeeklyPriceCl(Cl):
    dest_class = WeeklyPrice

    def __init__(self):
        super().__init__()

    def add_df(self, df, session=session):
        """Function to add data to the database.

        args:
        ----
        df - pandas dataframe - the data to be added to the database
        session - sqla session:None - the db session object

        returns:
        ----
        None
        """
        if df.shape[0]:
            df = df[['date', 'open', 'high', 'low', 'close', 'change', 'volume', 'ticker_id']] \
                .drop_duplicates()
            _add_df(df, WeeklyPrice, session=session)

    def fetch(self,
              ticker_ids=[],
              from_date=None,
              to_date=None
              ):
        """Function to create a query to grab all sub-jobs from the offers db.
        Open sub-jobs are set to status_id 1.

        args:
        ----
        ticker_ids - list:[] - the ids of the records to be extracted

        returns:
        ----
        sqlalchemy query 
        """
        query = session.query(WeeklyPrice)
        if len(ticker_ids):
            query = query.filter(WeeklyPrice.ticker_id.in_(ticker_ids))
        if from_date:
            query = query.filter(WeeklyPrice.date >= from_date)
        if to_date:
            query = query.filter(WeeklyPrice.date <= to_date)
        return query

    def fetch_latest(self,
                     session,
                     ticker_ids=[],
                     from_date=None,
                     to_date=None
                     ):
        """Function to get that last entry for each item

        args:
        ----
        session - sqla session
        ticker_ids - list:[] - the ids of the records to be extracted
        from_date - dtatime:None - the min date for filtering records
        to_date - dtatime:None - the max date for filtering records

        returns:
        ----
        sqla query
        """
        # create the sub-query
        subq = session.query(
            WeeklyPrice.ticker_id,
            func.max(WeeklyPrice.date).label("max_date")
        )
        # filter for dates
        if from_date:
            subq = subq.filter(WeeklyPrice.date >= from_date)
        if to_date:
            subq = subq.filter(WeeklyPrice.date <= to_date)
        # order the results
        subq = subq.order_by(WeeklyPrice.ticker_id, WeeklyPrice.date.desc()) \
            .group_by(WeeklyPrice.ticker_id) \
            .subquery("t2")
        # build the main query
        query = session.query(Ticker, subq.c.max_date) \
            .outerjoin(
                subq,
                subq.c.ticker_id == Ticker.id
        )
        # filter on ticker ids wanted
        if len(ticker_ids):
            query = query.filter(Ticker.id.in_(ticker_ids))
        return query

    def remove(self,
               ids=[],
               ticker_ids=[],
               from_date=None,
               to_date=None,
               del_all=False
               ):
        """Function to delete records from the weekly prices table.

        args:
        ----
        ids - list:[] - the ids of the records to be extracted
        ticker_ids - list:[] - the ids of the records to be extracted
        from_date - dtatime:None - the min date for filtering records
        to_date - dtatime:None - the max date for filtering records
        del_all - bool:False - safety to prevet deleting the whole table

        returns:
        ----
        sqlalchemy query 
        """
        try:
            # Preform check to prevent del_all
            if not del_all and not len(ids) and not len(ticker_ids) and not from_date and not to_date:
                logging.warning(
                    "Delete not performed as no attributes given and del_all is False")
                return False
            query = session.query(WeeklyPrice)
            if len(ids):
                query = query.filter(WeeklyPrice.id.in_(ids))
            if len(ticker_ids):
                query = query.filter(WeeklyPrice.ticker_id.in_(ticker_ids))
            if from_date:
                query = query.filter(WeeklyPrice.date >= from_date)
            if to_date:
                query = query.filter(WeeklyPrice.date <= to_date)
            query.delete(synchronize_session=False)
            session.commit()
            return True
        except:
            return False


ticker = TickerCl()
daily_price = DailyPriceCl()
weekly_price = WeeklyPriceCl()
