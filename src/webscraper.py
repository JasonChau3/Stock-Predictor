import yfinance as yf
import bs4 as bs
import pickle
import requests
import pandas as pd
from getTickers import *
import os

from pandas_datareader import data as pdr


def getData():
    yf.pdr_override()
    dow = save_dow_tickers()
    print(dow)
        
    data = pdr.get_data_yahoo(dow,period = "6mo", group_by='ticker')


    #loops for each ticker and create a dataaframe out of it
    for tickers in dow:
        df= pd.DataFrame(data[tickers])
        print(df)
        #max min normalize each column
        df =(df-df.min())/(df.max()-df.min())
        df.to_excel('../data/dowJonesData/' +tickers +'.xlsx')

    sp500tic = save_sp500_tickers()
    print(sp500tic)

    data = pdr.get_data_yahoo(sp500tic,period = "6mo", group_by='ticker')

    #loops for each ticker and create a dataaframe out of it
    for tickers in sp500tic:
        df= pd.DataFrame(data[tickers])
        df = (df-df.min())/(df.max()-df.min())
        df.to_excel('../data/SP500Data/' +tickers +'.xlsx')

    tickerdf = pd.DataFrame(sp500tic,columns=['ticker'])
#sectordf = pd.DataFrame(sp50Industry,columns=['industry'])

#tickerandsector = pd.concat([tickerdf, sectordf], axis=1,join = tickerdf.index)
#tickerandsector.to_excel('./SP500Data/Sp500Industries.xlsx')
