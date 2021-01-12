import yfinance as yf
import bs4 as bs
import pickle
import requests
from getTickers import *


dow = save_dow_tickers()
print(dow)
    
data = yf.download(dow,period = "6mo", group_by='ticker')
print(data)

