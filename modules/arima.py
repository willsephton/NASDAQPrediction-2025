import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

def arimaFunction(specficStock, chosen_ticker):
    # Fit ARIMA model (simple non-auto)
    model = ARIMA(specficStock, order=(2, 1, 2))
    model_fit = model.fit()
    
    # Forecast next 10 periods
    n_periods = 10
    forecast = model_fit.forecast(steps=n_periods)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(specficStock.index, specficStock, label='Original Data')
    plt.plot(pd.date_range(start=specficStock.index[-1], periods=n_periods + 1, freq='M')[1:], forecast, color='red', label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'ARIMA Forecast for {chosen_ticker}')
    plt.legend()
    st.pyplot(plt)
