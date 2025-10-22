from modules.dataRetrival import (getTickers, gatherStockDataPCAandKMeans, gatherStockDataCorrelationEDA, gatherStockDataForProphet)
from modules.pcaAndKMeans import (pcaFunction, kmeansFunction)
from modules.correlation import (correlationOutput)
from modules.eda import (stockPricesOverTime, distributionOfStockPricesBox, distributionOfStockPricesHistogram, monthlyStockLine, monthlyStockHistogram, weeklyStockBox)
from modules.prophetPrediction import (prophetFunction)
from modules.lstm import (lstm_stocks)
from modules.linearRegression import (linearRegressionFunction)
from modules.arima import (arimaFunction)

import streamlit as st

# ! StreamLit Configs

st.set_page_config(page_title="NASDAQ Stock Analysis", page_icon="📈")
st.title("📊 NASDAQ 100 Stock Prediction and Forecasting Dashboard")
st.markdown("Created by **Will Sephton** — BSc Project")

# ! Loads Tickers

@st.cache_data(show_spinner=False)
def load_tickers():
    return getTickers()

tickers = load_tickers()

# ! Sidebar

st.sidebar.header("🔧 Options")
chosen_ticker = st.sidebar.selectbox("Select a Ticker", tickers)
option = st.sidebar.selectbox(
    "Select a Module",
    (
        "🏠 Home",
        "📉 PCA and KMeans Clustering",
        "🔗 Correlation Analysis",
        "📈 Exploratory Data Analysis (EDA)",
        "🔮 Prophet Forecast",
        "🧮 ARIMA Forecast",
        "🧠 LSTM Forecast",
        "📏 Linear Regression Forecast"
    )
)

# ! Home Tab

if option == "🏠 Home":
    st.subheader("Welcome to the NASDAQ Forecasting Dashboard!")
    st.write("""
    This application allows you to analyze and predict stock data from the **NASDAQ 100** using multiple techniques:
    
    - **PCA and KMeans**: Dimensionality reduction and clustering  
    - **Correlation Analysis**: Explore how stocks move together  
    - **EDA**: Gain insights into stock behavior  
    - **Prophet, ARIMA, LSTM, Linear Regression**: Predict future prices  
    """)
    st.info("📅 All data is pulled in real-time (past 1 year). Model results depend on the latest market data.")

# ! PCA and K-Means

elif option == "📉 PCA and KMeans Clustering":
    st.subheader("PCA & KMeans Clustering on NASDAQ 100")
    with st.spinner("Loading data and running PCA + KMeans..."):
        st.subheader('Dataset after PCA and Kmeans clustering')
        st.write("The cluster label is listed on the far right of the dataset")
        PCAandKmeansData = gatherStockDataPCAandKMeans(tickers)
        kmeansData = pcaFunction(PCAandKmeansData)
        kmeansFunction(kmeansData, PCAandKmeansData)
    st.success("✅ PCA and KMeans completed successfully!")

# ! Correlation

elif option == "🔗 Correlation Analysis":
    st.subheader(f"Correlation Analysis for {chosen_ticker}")
    with st.spinner("Calculating correlations..."):
        correlationData = gatherStockDataCorrelationEDA(tickers)
        correlationOutput(correlationData, chosen_ticker)
    st.success("✅ Correlation analysis complete!")

# ! EDA

elif option == "📈 Exploratory Data Analysis (EDA)":
    st.subheader(f"Exploratory Data Analysis for {chosen_ticker}")
    with st.spinner("Generating EDA visualizations..."):
        data = gatherStockDataCorrelationEDA(tickers)

        with st.expander("📊 Temporal Structure"):
            stockPricesOverTime(data, chosen_ticker)

        with st.expander("📦 Distribution of Stock Prices"):
            distributionOfStockPricesBox(data, chosen_ticker)
            distributionOfStockPricesHistogram(data, chosen_ticker)

        with st.expander("📅 Monthly and Weekly Insights"):
            monthlyStockHistogram(data, chosen_ticker)
            weeklyStockBox(data, chosen_ticker)

    st.success("✅ EDA complete!")

# ! Prophet

elif option == "🔮 Prophet Forecast":
    st.subheader(f"Prophet Forecast for {chosen_ticker}")
    with st.spinner("Running Prophet model..."):
        data = gatherStockDataForProphet(tickers)
        days = st.slider("Select Forecasting Period (days)", 1, 365, 30)
        prophetFunction(chosen_ticker, days, data)
    st.success("✅ Prophet forecast complete!")

# ! ARIMA

elif option == "🧮 ARIMA Forecast":
    st.subheader(f"ARIMA Forecast for {chosen_ticker}")
    with st.spinner("Fitting ARIMA model... this may take a moment ⏳"):
        data = gatherStockDataForProphet(tickers)
        specific_stock = data[chosen_ticker]
        arimaFunction(specific_stock, chosen_ticker)
    st.success("✅ ARIMA forecast complete!")

# ! LSTM

elif option == "🧠 LSTM Forecast":
    st.subheader(f"LSTM Forecast for {chosen_ticker}")
    with st.spinner("Training LSTM model... please wait ⏳"):
        data = gatherStockDataForProphet(tickers)
        lstm_stocks(chosen_ticker, data)
    st.success("✅ LSTM prediction complete!")

# ! Linear Regression

elif option == "📏 Linear Regression Forecast":
    st.subheader(f"Linear Regression Forecast for {chosen_ticker}")
    with st.spinner("Running Linear Regression model..."):
        data = gatherStockDataForProphet(tickers)
        list_of_dates = list(range(len(data)))
        stock_data = data[chosen_ticker]
        linearRegressionFunction(list_of_dates, stock_data, chosen_ticker, data)
    st.success("✅ Linear Regression forecast complete!")
