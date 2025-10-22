
# ! Buy/Sell Signals

def createSignals(forecast):
    buy_threshold = 0.1 # Buy threshold is +10%
    sell_threshold = -0.1  # Sell threshold is -10%
    
    buy_count = sum(forecast['yhat'].pct_change() > buy_threshold)
    sell_count = sum(forecast['yhat'].pct_change() < sell_threshold)
    
    if buy_count > sell_count:
        return 'Buy'
    else:
        return 'Sell'