import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ! Long Short-Term Memory
    
def createDatasetforLSTM(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def lstm_stocks(chosen_stock, stockData):
    keras = tf.keras
    Sequential = keras.models.Sequential
    Dense = keras.layers.Dense
    LSTM = keras.layers.LSTM
    dataFrame = stockData.reset_index()[chosen_stock]

    scaler = MinMaxScaler()
    dataFrame = scaler.fit_transform(np.array(dataFrame).reshape(-1, 1))
    train_size = int(len(dataFrame) * 0.65)
    test_size = len(dataFrame) - train_size
    train_data, test_data = dataFrame[0:train_size, :], dataFrame[train_size:len(dataFrame), :1]

    time_step = 10
    X_train, Y_train = createDatasetforLSTM(train_data, time_step)
    X_test, Y_test = createDatasetforLSTM(test_data, time_step)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(
        X_train, Y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        verbose=1,
        shuffle=False
    )

    y_pred = model.predict(X_test)

    plt.plot(Y_test, marker='.', label="true")
    plt.plot(y_pred, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.legend()
    st.pyplot(plt)

    fig_prediction = plt.figure(figsize=(10, 8))
    plt.plot(np.arange(0, len(Y_train)), Y_train, 'g', label="history")
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), Y_test, marker='.', label="true")
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), y_pred, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.legend()
    st.pyplot(plt)
