import mplfinance as mpf
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime

from .PredictionWindow import WindowGenerator, compile_and_fit


def forecast(data_path, days, plot_test=False):
    out_steps = int(days)

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df.Date, infer_datetime_format=True)
    df.sort_values(by=['Date'], ascending=True, inplace=True, ignore_index=True)

    n = len(df)

    if plot_test:
        last_date = (df[:df.shape[0] - out_steps])['Date'].iloc[-1]
    else:
        last_date = df['Date'].iloc[-1]
    last_date = last_date + datetime.timedelta(days=1)
    date_time = pd.to_datetime(df.pop('Date'), format='%d.%m.%Y %H:%M:%S')
    column_indices = {name: i for i, name in enumerate(df.columns)}

    if not plot_test:
        empty_data = []
        for i in range(0,out_steps):
            empty_data.insert(0, {'Open': 0.0,'High':0.0,'Low':0.0,'Close':0.0,'Volume':0.0,'Market Cap':0.0})
        df = df.append(pd.DataFrame(empty_data), ignore_index=True)
    train_df = df[:df.shape[0] - out_steps]
    val_df = train_df[int(train_df.shape[0] * 0.95):]
    test_df = df[df.shape[0] - 2 * out_steps:]

    num_features = df.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    multi_window = WindowGenerator(input_width=1,
                                   label_width=out_steps,
                                   shift=out_steps,
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df)

    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(out_steps * num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([out_steps, num_features])
    ])

    history = compile_and_fit(multi_lstm_model, multi_window)
    predicts = multi_lstm_model.predict(multi_window.test)
    predicts_data = predicts[-1]
    for x in range(0, predicts_data.shape[0]):
        predicts_data[x] = predicts_data[x] * train_std + train_mean
    predicts_data = predicts_data[::-1]
    test_data = test_df
    test_data = test_data * train_std + train_mean
    test_data.reset_index(drop=True, inplace=True)
    date_range = pd.date_range(start=last_date, periods=out_steps)
    candle_plot(predicts_data, test_data[test_data.shape[0] - out_steps:], date_range, plot_test)

    return predicts_data


def candle_plot(predicts_data, test_data, date_time, plot_test):
    if type(predicts_data) is not pd.DataFrame:
        if isinstance(predicts_data, np.ndarray):
            predicts_data = np.delete(predicts_data, [4, 5], 1)
            predicts_data = pd.DataFrame(predicts_data, columns=['Open', 'High', 'Low', 'Close'])
        else:
            raise TypeError("Expect predicts_data as numpy.ndarray or pandas.DataFrame.")

    predicts_data['Datetime'] = pd.to_datetime(date_time)
    predicts_data = predicts_data.set_index('Datetime')

    if type(test_data) is not pd.DataFrame:
        raise TypeError("Expect test_data as pandas.DataFrame.")

    if plot_test:
        test_data['Datetime'] = pd.to_datetime(date_time)
        test_data = test_data.set_index('Datetime')

    title = 'Dogecoin Price Prediction'
    mpf.plot(predicts_data, type='candle', style='charles', title=title, ylabel='Price [USD]')
    if plot_test:
        mpf.plot(test_data, type='candle', style='charles', title=title+'_test', ylabel='Price [USD]')
