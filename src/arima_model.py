#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Title: Time-Series Forecasting: Predicting Stock Prices Using An ARIMA Model
Author: Serafeim Loukas
Date: July 23, 2020
Availability: https://towardsdatascience.com/time-series-forecasting-predicting-stock-prices-using-an-arima-model-2e3b3080bd70
'''


# In[2]:


#basic libraries for data analysis and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#time series functions from statsmodels
#from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA


# In[3]:


def arima_acc(ticker_name):
    ticker_df = pd.read_excel('data/dowJonesData' + ticker_name + '.xlsx', parse_dates = ['Date'])#, index_col = 'Date')
    #date as index is nice for plotting
    ticker_df = ticker_df.set_index('Date')
    #sets frequency of date indices
    #aapl.index = aapl.index.to_period('D')
    #label is 1 if closing price is greater than opening price else 0
    ticker_df['Label'] = ticker_df.apply(lambda row: 1 if row['Open'] < row['Close'] else 0, axis=1)
    
    training_data = ticker_df[:int(len(aapl)*.7)]
    test_data = ticker_df[int(len(aapl)*.7):]
    training_data = training_data['Close'].values
    test_data = test_data['Close'].values
    
    history = [x for x in training_data]
    model_predictions = []
    for t in range(len(test_data)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[t]
        history.append(true_test_value)
    
    test_open_prices = ticker_df['Open'][int(len(aapl)*0.7):].values
    test_preds = []
    for i in range(len(model_predictions)):
        #if predicted increase, predict 1, else 0
        if model_predictions[i] > test_open_prices[i]:
            test_preds.append(1)
        else:
            test_preds.append(0)
    test_preds = np.array(test_preds)
    
    test_labels = ticker_df['Label'][int(len(aapl)*0.7):].values
    accuracy = np.mean(test_labels == test_preds)
    return accuracy


# In[ ]:




