import yfinance as yf
import bs4 as bs
import pickle
import requests
from getTickers import *


dow = save_sp500_tickers()
print(dow)
    

yahoo = yf.Ticker(dow[0])
print(yahoo)

