from prophet import Prophet
from modules.buySellSignals import(createSignals)
import numpy as np
import streamlit as st

# ! Prophet Prediction
    
def prophetFunction(chosen_ticker, days, prophetData):
    data = prophetData.reset_index().rename(columns={'Date': 'ds', chosen_ticker: 'y'})
    data['y'] = np.log(data['y'])
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )

    model.fit(data)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    signals = createSignals(forecast)
    st.write(f"Signal for the forecast period: {signals}")

    st.pyplot(model.plot(forecast))