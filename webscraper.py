import yfinance as yf
import bs4 as bs
import pickle
import requests
import pandas as pd
from getTickers import *
import os

from pandas_datareader import data as pdr

yf.pdr_override()
dow = save_dow_tickers()
print(dow)
    
data = pdr.get_data_yahoo(dow,period = "6mo", group_by='ticker')


for tickers in dow:
    df= pd.DataFrame(data[tickers])
    df.to_excel('./data' +tickers +'.xlsx')

