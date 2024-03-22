# Libraries used 
import datetime as dt
import numpy as np
import os
import pandas as pd
import pickle
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import History
from zlib import crc32

history = History()  # Ignore, it helps with model_data function


def pickle_dump(stocks_prices):
    """
    Pickles the given pandas DataFrame containing stock prices into a file named "stocks_prices_dataframe.pkl".

    Parameters:
        stocks_prices (pandas.DataFrame): DataFrame containing stock prices to be pickled.

    Returns:
        None
    """
    with open("stocks_prices_dataframe.pkl", "wb") as f:
        pickle.dump(stocks_prices, f)


def pickle_load(filename):
    """
    Unpickles and loads a pandas DataFrame from the specified file.

    Parameters:
        filename (str): Name of the file to unpickle.

    Returns:
        pandas.DataFrame: DataFrame containing the unpickled data.
    """
    with open(filename, "rb") as f:
        stocks_prices = pickle.load(f)
    return stocks_prices


def load_dataframe(years):
    """
    Loads stock price data either from a pickled file or downloads it online using the yfinance library.
    Returns a pandas DataFrame containing the stock prices and a list of tickers.

    Parameters:
        years (int): Number of years of historical data to load.

    Returns:
        Tuple[pandas.DataFrame, List[str]]: A tuple containing the pandas DataFrame of stock prices and a list of tickers.
    """
    if os.path.isfile("stocks_prices_dataframe.pkl"):
        stock_prices = pickle_load("stocks_prices_dataframe.pkl")
        tickers = stock_prices.columns.tolist()
    else:
        tickers = get_stockex_tickers()
        stock_prices = loaded_df(years=years, tickers=tickers)

    return stock_prices, tickers


def hashing_and_splitting(adj_close_df):
    """
    Splits the given DataFrame of adjusted close prices into training and testing sets based on checksum hashing.

    Parameters:
        adj_close_df (pandas.DataFrame): DataFrame containing adjusted close prices.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: A tuple containing the training and testing DataFrames.
    """
    checksum = np.array([crc32(v) for v in adj_close_df.index.values])
    test_ratio = 0.2
    test_indices = checksum < test_ratio * 2 ** 32
    return adj_close_df[~test_indices], adj_close_df[test_indices]


def get_stockex_tickers(link):
    """
    Retrieves ticker symbols from a Wikipedia page containing stock exchange information.

    Parameters:
        link (str): Link to the Wikipedia page containing stock exchange information.

    Returns:
        List[str]: List of ticker symbols.
    """
    tables = pd.read_html(link)
    df = tables[4]
    df.drop(['Company', 'GICS Sector', 'GICS Sub-Industry'], axis=1, inplace=True)
    tickers = df['Ticker'].values.tolist()
    return tickers


def loaded_df(years, tickers):
    """
    Downloads stock price data for the specified number of years and tickers using yfinance.
    Returns a pandas DataFrame and pickles the data.

    Parameters:
        years (int): Number of years of historical data to load.
        tickers (List[str]): List of ticker symbols.

    Returns:
        pandas.DataFrame: DataFrame containing downloaded stock price data.
    """
    stocks_dict = {}
    time_window = 365 * years
    start_date = dt.datetime.now() - dt.timedelta(time_window)
    end_date = dt.datetime.now()
    for i, ticker in enumerate(tickers):
        print('Getting {} ({}/{})'.format(ticker, i, len(tickers)))
        prices = yf.download(ticker, start=start_date, end=end_date)
        stocks_dict[ticker] = prices['Adj Close']

    stocks_prices = pd.DataFrame.from_dict(stocks_dict)
    pickle_dump(stocks_prices=stocks_prices)
    return stocks_prices


def clean_df(percentage, tickers, stocks_prices):
    """
    Cleans the DataFrame by dropping stocks with NaN values exceeding the given percentage threshold.

    Parameters:
        percentage (float): Percentage threshold for NaN values.
        tickers (List[str]): List of ticker symbols.
        stocks_prices (pandas.DataFrame): DataFrame containing stock prices.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    if percentage > 1:
        percentage = percentage / 100
    for ticker in tickers:
        nan_values = stocks_prices[ticker].isnull().values.any()
        if nan_values:
            count_nan = stocks_prices[ticker].isnull().sum()
            if count_nan > (len(stocks_prices) * percentage):
                stocks_prices.drop(ticker, axis=1, inplace=True)
    return stocks_prices


def xtrain_ytrain(adj_close_df):
    """
    Splits the DataFrame into training and testing sets, normalizes the data, and prepares it for LSTM model training.

    Parameters:
        adj_close_df (pandas.DataFrame): DataFrame containing adjusted close prices.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]: A tuple containing training and testing data along with the scaler.
    """
    split_index = int((len(adj_close_df)) * 0.80)
    train_set = pd.DataFrame(adj_close_df.iloc[0:split_index])
    test_set = pd.DataFrame(adj_close_df.iloc[split_index:])

    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit(train_set)
    training_set_scaled = sc.fit_transform(train_set)
    test_set_scaled = sc.transform(test_set)

    xtrain = []
    ytrain = []
    for i in range(60, training_set_scaled.shape[0]):
        xtrain.append(training_set_scaled[i - 60:i, 0])
        ytrain.append(training_set_scaled[i, 0])
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)
    xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))

    xtest = []
    ytest = []
    for i in range(20, test_set_scaled.shape[0]):
        xtest.append(test_set_scaled[i - 20:i, 0])
        ytest.append(test_set_scaled[i, 0])
    xtest, ytest = np.array(xtest), np.array(ytest)
    return xtrain, ytrain, xtest, ytest, sc


def lstm_model(xtrain, ytrain):
    """
    Builds and trains an LSTM model using the training data.

    Parameters:
        xtrain (np.ndarray): Input training data.
        ytrain (np.ndarray): Target training data.

    Returns:
        Sequential: Trained LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(xtrain.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add
