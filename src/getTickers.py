import bs4 as bs 
import pickle
import requests
import pandas as pd
import re

def save_dow_tickers():
    resp = requests.get('https://finance.yahoo.com/quote/%5EDJI/components/')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find("table", {"data-reactid": "9"})

    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.find('td').text
        tickers.append(ticker)
    
    return tickers

def save_sp500_tickers():
    data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    table = data[0]
    #sliced_table = table[1:]
    #header = table.iloc[0]
    #corrected_table = sliced_table.rename(columns=header)
    tickers = list(table[1:]['Symbol'])
    return tickers


print(save_sp500_tickers())
