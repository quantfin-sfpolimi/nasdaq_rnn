# Libraries used
import datetime as dt
import numpy as np
import os
import pandas as pd
import pickle
import yfinance as yf
from openbb import obb
from matplotlib import pyplot as plt
import seaborn
import matplotlib.colors
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import History
from zlib import crc32
import re

history = History()  # Ignore, it helps with model_data function

# file saving with pickling

<<<<<<<<< Temporary merge branch 1
=========

>>>>>>>>> Temporary merge branch 2
def pickle_dump(obj, filename):
    """
    Serialize the given object and save it to a file using pickle.

        Parameters:
        obj:
            anything, dataset or ML model
        filename: str
            The name of the file to which the object will be saved. If the filename
            does not end with ".pkl", it will be appended automatically.

        Returns:
        None
        """
        if not re.search("^.*\.pkl$", filename):
            filename += ".pkl"

        file_path = "./pickle_files/" + filename
        with open(file_path, "wb") as f:
            pickle.dump(self.obj, f)

    @staticmethod
    def pickle_load(filename):
        """
        Load a serialized object from a file using pickle.

        Parameters:
        filename: str
            The name of the file from which the object will be loaded. If the filename
            does not end with ".pkl", it will be appended automatically.

    Returns:
    obj: any Python object
        The deserialized object loaded from the file.
    """
    if not re.search("^.*\.pkl$", filename):
        filename += ".pkl"
<<<<<<<<< Temporary merge branch 1
=========

        file_path = "./pickle_files/" + filename

    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        return obj
    except FileNotFoundError:
        print("This file " + file_path + " does not exists")
        return None
>>>>>>>>> Temporary merge branch 2

    file_path = "./pickle_files/" + filename

<<<<<<<<< Temporary merge branch 1
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        return obj
    except FileNotFoundError:
        print("This file " + file_path + " does not exists")
        return None

def load_dataframe(years, filename):
    """
    Load a DataFrame of stock prices from a pickle file if it exists, otherwise create a new DataFrame.

    Parameters:
    years: list
        A list of years for which the stock prices are required.
    filename: str
        The name of the file containing the serialized DataFrame. If the filename
        does not end with ".pkl", it will be appended automatically.

    Returns:
    stock_prices: DataFrame
        A DataFrame containing stock prices for the given years.
    tickers: list
        A list of tickers representing the stocks in the DataFrame.
    """
    if not re.search("^.*\.pkl$", filename):
        filename += ".pkl"

    file_path = "./pickle_files/" + filename

    if os.path.isfile(file_path):
        stock_prices = pickle_load(filename)
        tickers = stock_prices.columns.tolist()
    else:
        tickers = get_stockex_tickers()
        stock_prices = loaded_df(years=years, tickers=tickers)
=========
def load_dataframe(years, filename, link, interval):
    """
    Load a DataFrame of stock prices from a pickle file if it exists, otherwise create a new DataFrame.

    Parameters:
    years: list
        A list of years for which the stock prices are required.
    filename: str
        The name of the file containing the serialized DataFrame. If the filename
        does not end with ".pkl", it will be appended automatically.

    Returns:
    stock_prices: DataFrame
        A DataFrame containing stock prices for the given years.
    tickers: list
        A list of tickers representing the stocks in the DataFrame.
    """
    if not re.search("^.*\.pkl$", filename):
        filename += ".pkl"

    file_path = "./pickle_files/" + filename

    if os.path.isfile(file_path):
        stock_prices = pickle_load(filename)
        tickers = stock_prices.columns.tolist()
    else:
        tickers = get_stockex_tickers(link=link)
        stock_prices = loaded_df(
            years=years, tickers=tickers, interval=interval)

    return stock_prices, tickers


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
    df.drop(['Company', 'GICS Sector', 'GICS Sub-Industry'],
            axis=1, inplace=True)
    tickers = df['Ticker'].values.tolist()
    return tickers


def loaded_df(years, tickers, interval):
    """
    Downloads stock price data for the specified number of years and tickers using yfinance.
    Returns a pandas DataFrame and pickles the data.

    Parameters:
        years (int): Number of years of historical data to load.
        tickers (List[str]): List of ticker symbols.
        interval (str): Time frequency of historical data to load with format: ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1W', '1M' or '1Q').

    Returns:
        pandas.DataFrame: DataFrame containing downloaded stock price data.
    """
    stocks_dict = {}
    time_window = 365 * years
    start_date = dt.date.today() - dt.timedelta(time_window)
    end_date = dt.date.today()
    for i, ticker in enumerate(tickers):
        print('Getting {} ({}/{})'.format(ticker, i, len(tickers)))
        prices = obb.equity.price.historical(
            ticker, start_date=start_date, end_date=end_date, provider="yfinance", interval=interval).to_df()
        stocks_dict[ticker] = prices['close']
>>>>>>>>> Temporary merge branch 2

    stocks_prices = pd.DataFrame.from_dict(stocks_dict)
    return stocks_prices

# cleaning dataframe


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


<<<<<<<<< Temporary merge branch 1
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


def loaded_df(years, tickers, interval):
    """
    Downloads stock price data for the specified number of years and tickers using yfinance.
    Returns a pandas DataFrame and pickles the data.

    Parameters:
        years (int): Number of years of historical data to load.
        tickers (List[str]): List of ticker symbols.
        interval (str): Time frequency of historical data to load with format: ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1W', '1M' or '1Q').

    Returns:
        pandas.DataFrame: DataFrame containing downloaded stock price data.
    """
    stocks_dict = {}
    time_window = 365 * years
    start_date = dt.date.today() - dt.timedelta(time_window)
    end_date = dt.date.today()
    for i, ticker in enumerate(tickers):
        print('Getting {} ({}/{})'.format(ticker, i, len(tickers)))
        prices = obb.equity.price.historical(ticker ,start_date = start_date, end_date=end_date, provider="yfinance", interval=interval).to_df()
        stocks_dict[ticker] = prices['close']

    stocks_prices = pd.DataFrame.from_dict(stocks_dict)
    pickle_dump(stocks_prices=stocks_prices)
    return stocks_prices


=========
>>>>>>>>> Temporary merge branch 2
def clean_df(percentage, tickers, stocks_prices):
    """
    Cleans the DataFrame by dropping stocks with NaN values exceeding the given percentage threshold.
    The cleaned DataFrame is pickled after the operation.

        Parameters:
        self
        percentage : float
            Percentage threshold for NaN values. If greater than 1, it's interpreted as a percentage (e.g., 5 for 5%).
        
        Returns:
        None
        """
        if percentage > 1:
            percentage = percentage / 100

        for ticker in self.tickers:
            nan_values = self.prices[ticker].isnull().values.any()
            if nan_values:
                count_nan = self.prices[ticker].isnull().sum()
                if count_nan > (len(self.prices) * percentage):
                    self.prices.drop(ticker, axis=1, inplace=True)

        self.prices.ffill(axis=1, inplace=True) 
        PickleHelper(obj=self.prices).pickle_dump(filename='cleaned_nasdaq_dataframe')
        return None

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
    model.add(LSTM(units=50, activation='relu',
                  return_sequences=True, input_shape=(xtrain.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add
