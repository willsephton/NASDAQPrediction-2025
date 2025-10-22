import yfinance as yf
import pandas as pd
import requests
import streamlit as st

# ! Data Retrival

@st.cache_data(show_spinner=False)
def getTickers():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/128.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    tables = pd.read_html(response.text)
    nasdaq_table = next(
        t for t in tables if any(c in t.columns for c in ["Ticker", "Symbol"])
    )

    ticker_col = "Ticker" if "Ticker" in nasdaq_table.columns else "Symbol"
    tickers = nasdaq_table[ticker_col].dropna().tolist()
    print(f"✅ Found {len(tickers)} tickers")
    return tickers

#with open("dataset/list_of_tickers.txt", "r") as file:
    #tickers = file.read().splitlines()

@st.cache_data(show_spinner=False)
def gatherStockDataPCAandKMeans(tickers):
    
    stockData = yf.download(tickers, period='1y', interval='1d', group_by='tickers') # Downloads the nasdaq stock data
    closeData = stockData.xs('Close', level=1, axis=1)

    closeData = closeData.T

    cleanData = closeData.fillna(method='ffill').fillna(method='bfill').dropna(axis=1, how='all')

    return cleanData

@st.cache_data(show_spinner=True)
def gatherStockDataCorrelationEDA(tickers):
    stockData = yf.download(tickers, period='1y', interval='1d', group_by='tickers') # Downloads the nasdaq stock data
    closeData = stockData.xs('Close', level=1, axis=1)

    removedRows = closeData.dropna(axis=1) #Cleans rows with empty cells

    return removedRows

@st.cache_data(show_spinner=True)
def gatherStockDataForProphet(tickers):
    stockData = yf.download(tickers, period='1y', interval='1d')['Close']
    return stockData