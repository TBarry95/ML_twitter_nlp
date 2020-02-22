# pip install tweepy
# pip install alpha_vantage
# pip install quandl

from alpha_vantage.timeseries import TimeSeries
import alpha_vantage
import tweepy
import csv
import json
import quandl
import Functions as fns
import pandas as pd


def get_data_alpha_v2(ticker):
    api_key = '1TKL74QWO8OFMHQM'
    ts = TimeSeries(key=api_key, output_format='json')
    raw_price_data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    return raw_price_data

def alpha_v_to_df(ticker_dict):
    dates = [x[0] for x in ticker_dict.items()]
    ticker = [ticker_dict for x in ticker_dict.items()]
    open = [x[1]['1. open'] for x in ticker_dict.items()]
    close  = [x[1]['4. close'] for x in ticker_dict.items()]
    high  = [x[1]['2. high'] for x in ticker_dict.items()]
    low  = [x[1]['3. low'] for x in ticker_dict.items()]
    volume  = [x[1]['5. volume'] for x in ticker_dict.items()]
    return pd.DataFrame({"Dates": dates, "Ticker": ticker, "Open_Price": open, "Close_Price": close,
                         "Daily_High": high, "Daily_Low": low, "Trade_Volume": volume})

spx_df = alpha_v_to_df(get_data_alpha_v2("SPX"))
djia_df = alpha_v_to_df(get_data_alpha_v2("DJIA"))

combined_df = spx_df.append(djia_df)