import bs4 as bs 
import pickle
import requests
import re

def save_sp500_tickers():
    resp = requests.get('https://finance.yahoo.com/quote/%5EDJI/components/')
    print(resp);
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find("table", {"data-reactid": "9"})

    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.find('td').text
        tickers.append(ticker)
    
    return tickers


