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

import scipy.stats as ss

history = History()  # Ignore, it helps with model_data function

class PickleHelper:
    def __init__(self, obj):
        self.obj = obj

    def pickle_dump(self, filename):
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
        obj: PickleHelper
            A PickleHelper object with the obj loaded from the file accessible through its .obj attribute 
        """
        if not re.search("^.*\.pkl$", filename):
            filename += ".pkl"

        file_path = "./pickle_files/" + filename

        try:
            with open(file_path, "rb") as f:
                pcklHelper = PickleHelper(pickle.load(f))
            return pcklHelper
        except FileNotFoundError:
            print("This file " + file_path + " does not exists")
            return None

def hashing_and_splitting(adj_close_stocks_dataframe):
    """
    Splits the given DataFrame of adjusted close dataframe into training and testing sets based on checksum hashing.

    Parameters:
        adj_close_stocks_dataframe (pandas.DataFrame): DataFrame containing adjusted close dataframe.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: A tuple containing the training and testing DataFrames.
    """
    checksum = np.array([crc32(v) for v in adj_close_stocks_dataframe.index.values])
    test_ratio = 0.2
    test_indices = checksum < test_ratio * 2 ** 32
    return adj_close_stocks_dataframe[~test_indices], adj_close_stocks_dataframe[test_indices]

class DataFrameHelper:
    def __init__(self, filename, link, years, interval):
        self.filename = filename
        self.link = link
        self.years = years
        self.interval = interval
        self.dataframe = []
        self.tickers = []

    #NOTE: FUNZIONE LOAD MODIFICATA, non ritorna piu nulla ma aggiorna direttamente self.dataframe e self.tickers
    def load(self):
        """
        Load a DataFrame of stock dataframe from a pickle file if it exists, otherwise create a new DataFrame.

        Parameters: Obj
            self

        Returns: None
        """
        if not re.search("^.*\.pkl$", self.filename):
            self.filename += ".pkl"

        file_path = "./pickle_files/" + self.filename

        if os.path.isfile(file_path):
            self.dataframe = PickleHelper.pickle_load(self.filename).obj
            self.tickers = self.dataframe.columns.tolist()
        else:
            self.tickers = self.get_stockex_tickers()
            self.dataframe = self.loaded_df()

        return None

    def get_stockex_tickers(self):
        """
        Retrieves ticker symbols from a Wikipedia page containing stock exchange information.

        Parameters:
            self

        Returns:
            List[str]: List of ticker symbols.
        """
        tables = pd.read_html(self.link)
        df = tables[4]
        df.drop(['Company', 'GICS Sector', 'GICS Sub-Industry'],
                axis=1, inplace=True)
        tickers = df['Ticker'].values.tolist()
        return tickers

    def loaded_df(self):
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
        # simo's login with obb platform credetial
        obb.account.login(email='simo05062003@gmail.com', password='##2yTFb2F4Zd9z')
        stocks_dict = {}
        time_window = 365 * self.years
        start_date = dt.date.today() - dt.timedelta(time_window)
        end_date = dt.date.today()
        for i, ticker in enumerate(self.tickers):
            print('Getting {} ({}/{})'.format(ticker, i, len(self.tickers)))
            dataframe = obb.equity.price.historical(
                ticker, start_date=start_date, end_date=end_date, provider="yfinance", interval=self.interval).to_df()
            stocks_dict[ticker] = dataframe['close']

        stocks_dataframe = pd.DataFrame.from_dict(stocks_dict)
        return stocks_dataframe

    def clean_df(self, percentage):
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
            nan_values = self.dataframe[ticker].isnull().values.any()
            if nan_values:
                count_nan = self.dataframe[ticker].isnull().sum()
                if count_nan > (len(self.dataframe) * percentage):
                    self.dataframe.drop(ticker, axis=1, inplace=True)

        self.dataframe.ffill(axis=1, inplace=True) 
        PickleHelper(obj=self.dataframe).pickle_dump(filename='cleaned_nasdaq_dataframe')

def xtrain_ytrain(adj_close_stocks_dataframe):
    """
    Splits the DataFrame into training and testing sets, normalizes the data, and prepares it for LSTM model training.

    Parameters:
        adj_close_stocks_dataframe (pandas.DataFrame): DataFrame containing adjusted close dataframe.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]: A tuple containing training and testing data along with the scaler.
    """
    split_index = int((len(adj_close_stocks_dataframe)) * 0.80)
    train_set = pd.DataFrame(adj_close_stocks_dataframe.iloc[0:split_index])
    test_set = pd.DataFrame(adj_close_stocks_dataframe.iloc[split_index:])

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
    model.add #FIXME: something weird happened here

class CorrelationAnalysis:
    def __init__(self, prices, tickers, start_datetime, end_datetime):
        self.prices = prices
        self.tickers = tickers 
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.df = None # use corr_df() output to initialize

    def plot_corr_matrix(self):
        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [[norm(-1), "red"],
                  [norm(-0.93), "lightgrey"],
                  [norm(0.93), "lightgrey"],
                  [norm(1), "green"]]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        plt.figure(figsize=(40, 20))
        seaborn.heatmap(self.df, annot=True, cmap=cmap)
        plt.show()
        

    # datetime format '2024-02-15 09:30:00'
    def get_correlated_stocks(self):
        corr_df = self.prices.loc[self.start_datetime:self.end_datetime].corr(method='pearson')
        corr_true_or_false = corr_df.abs().ge(0.92)
        corr_dict = {}

        for ticker in self.tickers:
            df = corr_true_or_false.loc[corr_true_or_false[ticker] == True]
            x = list(df.index)
            x.remove(ticker)
            corr_dict[ticker] = x

            if len(corr_dict[ticker]) == 0:
                del corr_dict[ticker]

        return corr_dict, list(corr_dict.keys())

    def corr_df(self, corr_stocks_dict, corr_stocks_list):
        corr_stocks_df = pd.DataFrame()

        for ticker in self.tickers:
            if ticker in corr_stocks_list:
                corr_stocks_df[ticker] = self.prices[ticker]

        PickleHelper(obj=corr_stocks_df).pickle_dump(filename='correlatedstocks')

        return corr_stocks_df
