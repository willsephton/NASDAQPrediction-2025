import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

# ! EDA
        
# * Temporal Structure
def stockPricesOverTime(stockData, chosen_ticker):
    
    chosen_stock_data = stockData[chosen_ticker]


    plt.plot(chosen_stock_data.index, chosen_stock_data, label=chosen_ticker)

    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel(chosen_ticker+' Stock Price')
    plt.title(chosen_ticker+' Stock Prices Over Time')
    plt.legend()  # Show legend with ticker labels
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    st.pyplot(plt)  # Display the plot


# * Visualize the distribution of observation

def distributionOfStockPricesBox(stockData, chosen_ticker):

    stockData = stockData[chosen_ticker]

    plt.figure(figsize=(8, 6))

    plt.boxplot(stockData, vert=False)
    plt.title('Box Plot - Distribution of Stock Prices')
    plt.xlabel('Stock Prices')
    plt.yticks([])
    plt.grid(axis='x')

    st.pyplot(plt)


def distributionOfStockPricesHistogram(stockData, chosen_ticker):

    stockData = stockData[chosen_ticker]

    plt.figure(figsize=(8, 6))


    # Plotting a histogram for the selected stock's prices
    plt.hist(stockData, bins=30, color='skyblue')
    plt.title(f'Histogram of {chosen_ticker} Stock Prices')
    plt.xlabel('Stock Prices')
    plt.ylabel('Frequency')

    st.pyplot(plt)


# * Investigate the change in distribution over intervals

def monthlyStockLine(stockData, chosen_ticker):
    
    stockData = stockData[chosen_ticker]
    stockData.index = pd.to_datetime(stockData.index)

    # Resample the data into monthly intervals and use mean
    stockDataMonthly = stockData.resample('M').mean()  



    plt.plot(stockDataMonthly.index, stockDataMonthly, label=chosen_ticker)

    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel(f'{chosen_ticker} Stock Price')
    plt.title(f'{chosen_ticker} Stock Prices Over Time')
    plt.legend()  # Show legend with ticker labels
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    st.pyplot(plt)

def monthlyStockHistogram(stockData, chosen_ticker):
    
    # Extracting the chosen ticker's data
    stockData = stockData[chosen_ticker]
    
    # Converting the index to a DateTimeIndex if it's not already in that format
    stockData.index = pd.to_datetime(stockData.index)

    # Resampling the data into monthly intervals and use mean
    stockDataMonthly = stockData.resample('M').mean()  

    plt.figure(figsize=(10, 6))

    # Plotting a histogram for the selected stock's monthly average prices
    plt.hist(stockDataMonthly, bins=30, color='skyblue')
    plt.title(f'Histogram of {chosen_ticker} Stock Monthly Average Prices')
    plt.xlabel('Monthly Average Stock Prices')
    plt.ylabel('Frequency')

    st.pyplot(plt)

def weeklyStockBox(stockData, chosen_ticker):

    stockData = stockData[chosen_ticker]
    weekly_data = stockData.resample('W').mean()

    # Plotting the box plot for weekly data
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=weekly_data.values)
    plt.xlabel('Weeks')
    plt.ylabel('Closing Price')
    plt.title(f'Distribution of Weekly Closing Prices for {chosen_ticker}')
    st.pyplot(plt)