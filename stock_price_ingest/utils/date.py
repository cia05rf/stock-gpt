import datetime as dt
import pandas as pd
from numpy import ones, zeros, concatenate, datetime64

#Convert datetime strings
def conv_dt(v, date_or_time=None, tz=None):
    """Function for converting datetime strings to 
    dates, times, or datetimes

    args:
    ----
    v - str - the value to be converted in the format "%Y-%m-%d %H:%M:%S"
    date_or_time - str:None - must be 'date', 'time', or 'datetime'
    tz - pytz object - a timezne object

    returns:
    ----
    date, time, datetime, or None
    """
    if date_or_time == "date":
        if tz:
            return tz.localize(dt.datetime.strptime(v[:10],"%Y-%m-%d"))
        else:
            return dt.datetime.strptime(v[:10],"%Y-%m-%d")
    elif date_or_time == "time":
        if tz:
            return tz.localize(dt.datetime.strptime(v[11:19],"%H:%M:%S"))
        else:
            return dt.datetime.strptime(v[11:19],"%H:%M:%S")
    elif date_or_time == "datetime":
        if tz:
            return tz.localize(dt.datetime.strptime(v[:19],"%Y-%m-%d %H:%M:%S"))
        else:
            return dt.datetime.strptime(v[:19],"%Y-%m-%d %H:%M:%S")
    elif date_or_time == "short_date":
        if tz:
            return tz.localize(dt.datetime.strptime(v[:12],"%b %d, %Y"))
        else:
            return dt.datetime.strptime(v[:12],"%b %d, %Y")
    else:
        return None

def calc_date_window(st_date:int, en_date:int):
    #Establish the day ref of the dates compared to 01/01/1970
    ep_date = dt.datetime(1970, 1, 1)
    st_days = (st_date - ep_date).days
    en_days = (en_date - ep_date).days
    return [st_days*86400, en_days*86400]

#Create a list of time intervals to be used with 140 days in each item
def create_sec_ref_li(st_date:int, en_date:int, days:int=140):
    """Function to create a list of time intervals with a set number of days
    within each interval.

    Used for webscrape calls (too big a time frame causes the scrape to crash).

    args:
    ------
    st_date - int - the first date in the time period you with to scrape (inclusive)
    en_date - int - the first date in the time period you with to scrape (inclusive)
    days - int:140 - the time window size you wish returned in days

    returns:
    ------
    list of tuples (datetime, datetime)

    """
    #make sure the dates are different
    if st_date == en_date:
        return []
    #Establish the day ref of the dates compared to 01/01/1970
    ep_date = pd.to_datetime(dt.datetime(1970, 1, 1), errors='coerce')
    en_date = pd.to_datetime(en_date, errors='coerce')
    st_date = pd.to_datetime(st_date, errors='coerce')
    st_days = (st_date - ep_date).days
    en_days = (en_date - ep_date).days
    #Loop adding to a list until reaching 0
    sec_ref_li = []
    while en_days > st_days:
        if en_days - days > st_days:
            sec_ref_li.append([(en_days - days)*86400, en_days*86400])
        else:
            sec_ref_li.append([st_days*86400, en_days*86400])        
        en_days += -days
    return sec_ref_li

def calc_en_date(in_date=dt.datetime.today()):
    #Establish the end date for scrapping
    en_date = in_date + dt.timedelta(days=1)
    en_date = dt.datetime(en_date.year, en_date.month, en_date.day, 0, 0, 0)
    #Match to sat if sunday
    if en_date.weekday() == 6:
        en_date = en_date - dt.timedelta(days=(en_date.weekday() - 5))
    return en_date

def calc_wk_st_date(in_date=dt.datetime.today()):
    in_date = dt.datetime(in_date.year, in_date.month, in_date.day, 0, 0, 0)
    #Get monday date
    wk_st_date = in_date - dt.timedelta(days=in_date.weekday())
    return wk_st_date

def calc_st_date(in_date=dt.datetime.today()):
    #Add 1 day to st_date
    st_date = in_date + dt.timedelta(days=1)
    return st_date

def create_full_year_days(year, from_date=None, to_date=None):
    st_date = dt.datetime(year, 1, 1)
    if from_date:
        st_date = st_date.replace(month=from_date.month, day=from_date.day)
    delta = dt.timedelta(days=1)
    year_dates = []
    cur_date = st_date
    while cur_date.year == year:
        year_dates.append(cur_date)
        cur_date = cur_date + delta
        if to_date and cur_date >= to_date:
            break
    return year_dates