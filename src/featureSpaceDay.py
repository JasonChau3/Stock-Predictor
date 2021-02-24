import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
pd.set_option('display.max_columns', None)

import yfinance as yf
import bs4 as bs
import pickle
import requests
import pandas as pd
import sys
import os

from itertools import chain
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\src")
from  getTickers import *
import os
from pandas_datareader import data as pdr


''' 
This method would give you the feature Space of all the stocks for a certain number of days for 
and starting at at a certain day.For example, for day 3 and 3 numDays, we will give a feature
space of all the stocks on the third day + 3 more days so from day three through day six. It will
also return a label specifying a 1, if on the next day the Closing price closes higher than on the
opening price, and a -1 if the opening is higher.

Parameters:
day (int): The starting day of the feature space
numDays (int) : The number of days that we would use on top of the specified day

Return:
featureSpace (DataFrame) : Columns will be the numDays * 4 and the len(rows) will have the number of 
stocks in the datasets.
'''

#numDays for 4 columns * numDays
#col would have open high low, volume
#day is the day at e.g day (0)
#return featurespace and the label( label is 0 if close is lower that the open on the next day, 
#label is 1 if close is higher than open on the next day)
def featureDaySpace(numDays, day):
    labels = [] # array of 30 labels for each stock
    
    dow = save_dow_tickers()
    featureVals = []

    data = pdr.get_data_yahoo(dow,period = "6mo", group_by='ticker')
    for x in dow:
        df = data[x]
        df = df.iloc[1:]
        df = (df-df.min())/(df.max()-df.min())
        #get the labels by seeing if the next day closing > opening 
        dayAfterRow = df.iloc[day+numDays]
        #open price at index 0 
        openPrice = dayAfterRow.iloc[0]
        #close at index 3
        closePrice = dayAfterRow.iloc[3]

        if closePrice > openPrice:
            labels.append(1)
        else:
            labels.append(0)
            
        #get the feature space
        df = df.drop(columns = ['Adj Close','Volume'])


        featureVals.append(list(chain.from_iterable(df.iloc[day:day+numDays-1].values.tolist())))

    tempCol = ['Open','High','Low','Volume']
    col = []
    for x in range(numDays-1):
        for y in tempCol:
            col.append(y + ' Day ' + str(x) )
    
    featurespace = pd.DataFrame(featureVals, columns = col) 
    
    return featurespace, labels
