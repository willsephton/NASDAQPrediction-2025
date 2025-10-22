import streamlit as st

# ! Colleration

def topTenCorrelation(stockData, chosen_ticker):
        st.write(f"Top Ten Correlation for {chosen_ticker}")
        correlated = stockData.corrwith(stockData[chosen_ticker])

        highest11 = correlated.nlargest(11)
        highest10 = highest11.iloc[1:]  # Using iloc to select rows
        st.write(highest10)
        return highest10
    
def bottomTenCorrelation(stockData, chosen_ticker):
        st.write(f"Bottom Ten Correlation for {chosen_ticker}")
        correlated = stockData.corrwith(stockData[chosen_ticker])
        lowest10 = correlated.nsmallest(10)
        #lowest10 = lowest116.iloc[1:]  # Using iloc to select rows
        st.write(lowest10)
        return lowest10

def correlationOutput(stockData, chosen_ticker):
    topTenCorrelation(stockData, chosen_ticker)

    bottomTenCorrelation(stockData, chosen_ticker)
