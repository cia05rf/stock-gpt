"""Functions for updating data in the prices database"""
import re
import pandas as pd

from utils.data import overlap

from database.models import Session as session


def _update_df(df: pd.DataFrame, DestClass, session=session):
    """Function for updating records from a dataframe"""
    # Get table columns
    tab_cols = DestClass.__table__.columns
    # Remove table name prefix
    tab_name = DestClass.__table__.name
    tab_cols = [re.sub(fr"^{tab_name}\.", "", str(c)) for c in tab_cols]
    cols = overlap([tab_cols, df.columns])
    session.bulk_update_mappings(DestClass, df[cols].to_dict(orient="records"))
    session.commit()


def _add_df(df, DestClass, fields=[], session=session):
    """Generic function for adding to a table from a dataframe.

    args:
    ----
    df - pandas dataframe - the records to be added to the database 
    DestClass - sqla table class - the class of the receiving table
    session - sqla session:None - the offers db session object

    returns:
    ----
    None
    """
    # Get table columns
    tab_cols = DestClass.__table__.columns
    # Remove table name prefix
    tab_name = DestClass.__table__.name
    tab_cols = [re.sub(fr"^{tab_name}\.", "", str(c)) for c in tab_cols]
    cols = overlap([tab_cols, df.columns])
    if len(fields) > 0:
        cols = overlap([cols, fields])
    df = df[cols]
    objects = [DestClass(**r) for _, r in df.iterrows()]
    session.bulk_save_objects(objects)
    session.commit()

# Converting query to dataframe


def sqlaq_to_df(self, query, session=session, limit=None):
    out_df = pd.read_sql(query.statement, con=session.bind)
    out_df = out_df.iloc[:limit] if limit < out_df.shape[0] else out_df
    return out_df


def sqlaq_to_df_first(query, session=session):
    out_df = pd.read_sql(query.statement, con=session.bind)
    return out_df.iloc[0]
