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
    df.to_excel('./dowJonesData/' +tickers +'.xlsx')

print('helo');
sp500tic = save_sp500_tickers()
print(sp500tic)

data = pdr.get_data_yahoo(sp500tic,period = "6mo", group_by='ticker')

for tickers in sp500tic:
    df= pd.DataFrame(data[tickers])
    df.to_excel('./SP500Data/' +tickers +'.xlsx')

tickerdf = pd.DataFrame(sp500tic,columns=['ticker'])
#sectordf = pd.DataFrame(sp50Industry,columns=['industry'])

#tickerandsector = pd.concat([tickerdf, sectordf], axis=1,join = tickerdf.index)
#tickerandsector.to_excel('./SP500Data/Sp500Industries.xlsx')
