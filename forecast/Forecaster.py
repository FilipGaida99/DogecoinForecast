import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from .PredictionWindow import WindowGenerator, compile_and_fit


def forecast(data_path, days, plot_test=False):
    out_steps = int(days)

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df.Date, infer_datetime_format=True)
    df.sort_values(by=['Date'], ascending=True, inplace=True, ignore_index=True)

    date_time = pd.to_datetime(df.pop('Date'), format='%d.%m.%Y %H:%M:%S')
    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    num_features = df.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    multi_window = WindowGenerator(input_width=out_steps,
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
    predicts_data = predicts
    sums = predicts_data[0]
    for x in predicts_data[1:]:
        sums += x
    predicts_data = sums / predicts.shape[0]
    predicts_data = predicts_data * train_std[3] + train_mean[3]
    if plot_test:
        plt.plot(predicts_data[:, 3], label="Predicted price", color='red')
        test_data = test_df['Close'] * train_std[3] + train_mean[3]
        test_data.reset_index(drop=True, inplace=True)
        test_data = test_data[:out_steps]
        plt.plot(test_data, label="Actual price", color='green')
        plt.xlabel('Days')
        plt.ylabel('Price(USD)')
        plt.title('Dogecoin Price Prediction')
        plt.legend(loc='best')
        plt.show()

    return predicts_data
