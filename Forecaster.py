import operator
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential


def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)


def preprocess(data_raw, seq_len, days):
    data = to_sequences(data_raw, seq_len)

    x_train = data[:data.shape[0] - days - 1, :-1, :]
    y_train = data[:data.shape[0] - days - 1, -1, :]

    x_predict = data[data.shape[0] - days:, :-1, :]

    return x_train, y_train, x_predict


def forecast(data_path, days):
    BATCH_SIZE = 64
    SEQ_LEN = 100
    DROPOUT = 0.2
    WINDOW_SIZE = SEQ_LEN - 1

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df.Date, infer_datetime_format=True)
    base = df['Date'].max() + datetime.timedelta(days=1)

    dfa = pd.DataFrame({'Date': [base + datetime.timedelta(days=x) for x in range(days)]})
    df = df.append(dfa, ignore_index=True)
    df.sort_values(by=['Date'], ascending=True, inplace=True, ignore_index=True)

    scaler = MinMaxScaler()

    close_price = df.Close.values.reshape(-1, 1)

    scaled_close = scaler.fit_transform(close_price)

    #scaled_close = scaled_close[~np.isnan(scaled_close)]
    scaled_close = scaled_close.reshape(-1, 1)

    x_train, y_train, x_predict = \
        preprocess(scaled_close, SEQ_LEN, days)

    num_features = df.shape[1]
    model = keras.Sequential()

    model.add(Bidirectional(
        LSTM(WINDOW_SIZE, return_sequences=True),
        input_shape=(WINDOW_SIZE, x_train.shape[-1])
    ))
    model.add(Dropout(rate=DROPOUT))

    model.add(Bidirectional(
        LSTM((WINDOW_SIZE * 2), return_sequences=True)
    ))
    model.add(Dropout(rate=DROPOUT))

    model.add(Bidirectional(
        LSTM(WINDOW_SIZE, return_sequences=False)
    ))

    model.add(Dense(units=1))

    model.add(Activation('linear'))

    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.1
    )

    y_hat = model.predict(x_predict)
    y_hat_inverse = scaler.inverse_transform(y_hat)
    return y_hat_inverse
