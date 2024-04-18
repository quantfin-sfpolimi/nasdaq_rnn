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
        obb.account.login(email='simo05062003@gmail.com', password='##2yTFb2F4Zd9z')

        if not re.search("^.*\.pkl$", self.filename):
            self.filename += ".pkl"

        file_path = "./pickle_files/" + self.filename

        if os.path.isfile(file_path):
            self.dataframe = PickleHelper.pickle_load(self.filename).obj
            self.dataframe.info() #FIXME: testing
            self.tickers = self.dataframe.columns.tolist()
        else:
            self.tickers = self.get_stockex_tickers()
            self.dataframe = self.loaded_df()
            self.dataframe.info()

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
        #fml this doesn't work if i have consecutive days
        self.dataframe = self.dataframe.drop(index=[876, 958, 1031, 1032]) #FIXME: workin on it 

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
    """
    A class for performing correlation analysis on stock data.
    
    Attributes:
        dataframe (pandas.DataFrame): The DataFrame containing the stock data.
        tickers (list): List of ticker symbols representing the stocks.
        start_datetime (str): Start date and time of the data in 'YYYY-MM-DD HH:MM:SS' format.
        end_datetime (str): End date and time of the data in 'YYYY-MM-DD HH:MM:SS' format.
        corrvalues (np.ndarray): Array containing correlation coefficients.
        pvalues (np.ndarray): Array containing p-values.
        winner (list): A list containing ticker symbols corresponding to the pair with the maximum correlation coefficient.
    """

    def __init__(self, dataframe, tickers, start_datetime, end_datetime):
        """
        Initialize the CorrelationAnalysis object.
        
        Args:
            dataframe (pandas.DataFrame): The DataFrame containing the stock data.
            tickers (list): List of ticker symbols representing the stocks.
            start_datetime (str): Start date and time of the data in 'YYYY-MM-DD HH:MM:SS' format.
            end_datetime (str): End date and time of the data in 'YYYY-MM-DD HH:MM:SS' format.
        """
        self.dataframe = dataframe
        self.tickers = tickers 
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.corrvalues = None
        self.pvalues = None
        self.winner = None

    def get_correlated_stocks(self):
        """
        Calculate correlation coefficients and p-values for the given stocks within a given time period.
        
        Returns:
            None
        """
        corr_values = np.zeros([len(self.tickers), len(self.tickers)])
        pvalue_array = np.zeros([len(self.tickers), len(self.tickers)])
        for i in range(len(self.tickers)):
            for j in range(len(self.tickers)):
                vals_i = self.dataframe[self.tickers[i]].to_numpy()
                vals_j = self.dataframe[self.tickers[j]].to_numpy()
                r_ij, p_ij = ss.stats.pearsonr(vals_i, vals_j)
                corr_values[i, j] = r_ij
                pvalue_array[i, j] = p_ij
                
        self.corrvalues = corr_values
        self.pvalues = pvalue_array
        PickleHelper(self.corrvalues).pickle_dump('correlationvalues_array')
        PickleHelper(self.pvalues).pickle_dump('pvalues_array')

    def plot_corr_matrix(self):
        """
        Plot the correlation matrix heatmap for the given DataFrame.
        
        Returns:
            None
        """
        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [[norm(-1), "red"],
                  [norm(-0.93), "lightgrey"],
                  [norm(0.93), "lightgrey"],
                  [norm(1), "green"]]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        plt.figure(figsize=(40, 20))
        seaborn.heatmap(pd.DataFrame(self.corrvalues, columns=self.tickers, index=self.tickers), annot=True, cmap=cmap)
        plt.show()

    def corr_stocks_pair(self):
        """
        Identify the pair of stocks with the maximum correlation coefficient and save it to a pickle file.
        
        Returns:
            None
        """
        corr_values_filtered = np.where(self.pvalues > 0.05, self.corrvalues, np.nan)
        min_corr = np.nanmin(corr_values_filtered)
        tmp_arr = corr_values_filtered.copy()
        for i in range(len(tmp_arr)):
            tmp_arr[i, i] = 0
        max_corr = np.nanmax(tmp_arr)
        max_indexes = np.where(self.corrvalues == max_corr)
        max_pair = [self.tickers[max_indexes[0][0]], self.tickers[max_indexes[0][1]]]

        corr_order = np.argsort(tmp_arr.flatten())
        corr_num = corr_order[-1]
        max_pair = [self.tickers[corr_num // len(self.tickers)], self.tickers[corr_num % len(self.tickers)]]
        self.winner = max_pair
        print(max_pair)
        PickleHelper(self.winner).pickle_dump('df_maxcorr_pair')
        plt.figure(figsize=(40,20))
        plt.plot(self.dataframe[max_pair[1]])
        plt.plot(self.dataframe[max_pair[0]])
        plt.show
