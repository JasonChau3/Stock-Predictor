#!/usr/bin/env python
# coding: utf-8

# In[2]:


#basic libraries for data analysis and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#time series functions from statsmodels
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA


# In[3]:


aapl = pd.read_excel('AAPL.xlsx', parse_dates = ['Date'])#, index_col = 'Date')
#date as index is nice for plotting
aapl = aapl.set_index('Date')
#sets frequency of date indices
#aapl.index = aapl.index.to_period('D')
#label is 1 if closing price is greater than opening price else 0
aapl['Label'] = aapl.apply(lambda row: 1 if row['Open'] < row['Close'] else 0, axis=1)
aapl.head()


# In[11]:


aapl['Open'].plot()
aapl['Close'].plot()
plt.title('AAPL Opening and Closing Stock Price')
plt.xlabel('Date')
plt.ylabel('Price per Share')
plt.legend()
plt.show()


# In[12]:


#ideally an ARIMA model can predict a stationary time series that has unit root
stat_test = adfuller(aapl['Close'])
#from the p-value and critical value, the stock price is not stationary
print(f"Test statstic: {stat_test[0]} \np-val: {stat_test[1]} \nCritical value at 10% significance: {stat_test[4]['10%']}")


# In[13]:


#shows that about first 10 lags are significant
autocorrelation_plot(aapl['Close'])
plt.show()


# In[14]:


#take log of time series to make more stationary
aapl_log = np.log(aapl['Close'])
#shift series by one to get difference
aapl_log_shift = aapl_log - aapl_log.shift()
#first result is NA when you shift, makes ARIMA model throw error
aapl_log_shift = aapl_log_shift.dropna()
aapl_log_shift.plot()
plt.show()


# In[17]:


#use simpler model to load faster, using 10 lags is overkill
#maybe change parameters later through grid search
model = ARIMA(aapl_log, order=(5,1,0))
#shift data by 1
results = model.fit(disp=-1)
#plot arima and stationary series
plt.plot(aapl_log_shift)
plt.plot(results.fittedvalues)
plt.legend()
plt.show()


# In[18]:


#copy arima to its own series
shifted_predictions = pd.Series(results.fittedvalues, copy=True)
#get cumulative sum of shifted predictions
shifted_predictions = shifted_predictions.cumsum()
#put logged predictions into its own series
logged_predictions = pd.Series(aapl_log.iloc[0], index=aapl_log.index)
#add the two series together
logged_predictions = logged_predictions.add(shifted_predictions, fill_value=0)
#exponentiate to undo log
arima_predictions = np.exp(logged_predictions)
#plot data
plt.plot(aapl['Close'])
plt.plot(arima_predictions)
plt.show()


# In[19]:


predictions = arima_predictions.diff().apply(lambda x: 1 if x > 0 else 0)
predictions.head()


# In[20]:


test_predictions = predictions.iloc[int(aapl.shape[0] * .8):]
test_labels = aapl.iloc[int(aapl.shape[0] * .8):]['Label']


# In[24]:


accuracy = np.mean(test_predictions == test_labels)
accuracy


# In[ ]:




