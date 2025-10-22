import numpy as np
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ! Linear Regression

def linearRegressionFunction(listOfDates, chosenStockData, chosen_ticker, linearRegData):
    # Convert input data into numpy arrays and reshape them
    listOfDates = np.asanyarray(listOfDates)  # Convert list of dates to a NumPy array
    chosenStockData = np.asanyarray(chosenStockData)  # Convert stock data to a NumPy array
    listOfDates = np.reshape(listOfDates, (len(listOfDates), 1))  # Reshape date array into (length, 1)
    chosenStockData = np.reshape(chosenStockData, (len(chosenStockData), 1))  # Reshape stock data array into (length, 1)

    # Attempt to load the previously saved model to evaluate its performance
    try:
        pickle_in = open("prediction.pickle", "rb")
        reg = pickle.load(pickle_in)
        xtrain, xtest, ytrain, ytest = train_test_split(listOfDates, chosenStockData, test_size=1)
        best = reg.score(ytrain, ytest)  # Evaluate model accuracy using test data
    except:
        pass  # If loading the model fails, proceed without errors

    # Initialize the variable to hold the best accuracy achieved
    best = 0

    # Train the model iteratively multiple times to find the best accuracy
    for z in range(100):
        xtrain, xtest, ytrain, ytest = train_test_split(listOfDates, chosenStockData, test_size=0.80)
        reg = LinearRegression().fit(xtrain, ytrain)  # Fit a Linear Regression model
        accuracy = reg.score(xtest, ytest)  # Calculate the accuracy of the model
        # Check if the current accuracy is better than the previous best accuracy
        if accuracy > best:
            best = accuracy  
            with open('prediction.pickle', 'wb') as f:
                pickle.dump(reg, f)  # Save the best model using pickle
            print(accuracy)  # Print the current best accuracy

    # Load the best model obtained during training
    pickle_in = open("prediction.pickle", "rb")
    reg = pickle.load(pickle_in)

    # Evaluate the average accuracy of the best model over multiple iterations
    mean = 0
    for i in range(10):
        msk = np.random.rand(len(linearRegData)) < 0.8
        xtest = listOfDates[~msk]
        ytest = chosenStockData[~msk]
        mean += reg.score(xtest, ytest)  # Calculate accuracy using test data

    print("Average Accuracy:", mean / 10)

    # Plot the actual and predicted stock prices
    plt.plot(xtest, ytest, color='blue', linewidth=1, label='Actual Stock Price') 
    plt.plot(xtest, reg.predict(xtest), color='red', linewidth=3, label='Predicted Stock Price')  
    plt.title(f"Linear Regression for {chosen_ticker}") 
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    st.pyplot(plt) 