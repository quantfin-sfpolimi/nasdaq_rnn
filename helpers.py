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

# file saving with pickling


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
    # Check if filename ends with ".pkl", if not add it
    if not re.search("^.*\.pkl$", filename):
        filename += ".pkl"

    file_path = "./pickle_files/" + filename
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


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

    file_path = "./pickle_files/" + filename

    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        return obj
    except FileNotFoundError:
        print("This file " + file_path + " does not exists")
        return None


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
        stock_prices = loaded_stocks_prices(
            years=years, tickers=tickers, interval=interval)

    return stock_prices, tickers


def get_stockex_tickers(link):
    """
    Retrieves ticker(tickers from a Wikipedia page containing stock exchange information.

    Parameters:
        link (str): Link to the Wikipedia page containing stock exchange information.

    Returns:
        List[str]: List of ticker(tickers.
    """
    tables = pd.read_html(link)
    stocks_prices = tables[4]
    stocks_prices.drop(['Company', 'GICS Sector', 'GICS Sub-Industry'],
            axis=1, inplace=True)
    tickers = stocks_prices['Ticker'].values.tolist()
    return tickers


def loaded_stocks_prices(years, tickers, interval):
    """
    Downloads stock price data for the specified number of years and tickers using yfinance.
    Returns a pandas DataFrame and pickles the data.

    Parameters:
        years (int): Number of years of historical data to load.
        tickers (List[str]): List of ticker(tickers.
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
        prices = obb.equity.price.historical(ticker, start_date=start_date, end_date=end_date, provider="yfinance", interval=interval) #FIXME: method doesn't exist?
        stocks_dict[ticker] = prices['close'] #FIXME: obb object not subscriptable

    stocks_prices = pd.DataFrame.from_dict(stocks_dict)
    return stocks_prices

# cleaning dataframe


def hashing_and_splitting(adj_close_stocks_prices):
    """
    Splits the given DataFrame of adjusted close prices into training and testing sets based on checksum hashing.

    Parameters:
        adj_close_stocks_prices (pandas.DataFrame): DataFrame containing adjusted close prices.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: A tuple containing the training and testing DataFrames.
    """
    checksum = np.array([crc32(v) for v in adj_close_stocks_prices.index.values])
    test_ratio = 0.2
    test_indices = checksum < test_ratio * 2 ** 32
    return adj_close_stocks_prices[~test_indices], adj_close_stocks_prices[test_indices]


def clean_stocks_prices(percentage, tickers, stocks_prices):
    """
    Cleans the DataFrame by dropping stocks with NaN values exceeding the given percentage threshold.
    The cleaned DataFrame is pickled after the operation.

    Parameters:
    percentage : float
        Percentage threshold for NaN values. If greater than 1, it's interpreted as a percentage (e.g., 5 for 5%).
    tickers : List[str]
        List of ticker(tickers.
    stocks_prices : pandas.DataFrame
        DataFrame containing stock prices.

    Returns:
    pandas.DataFrame
        Cleaned DataFrame with NaN values exceeding the threshold removed.
    """
    if percentage > 1:
        percentage = percentage / 100

    for ticker in tickers:
        nan_values = stocks_prices[ticker].isnull().values.any()
        if nan_values:
            count_nan = stocks_prices[ticker].isnull().sum()
            if count_nan > (len(stocks_prices) * percentage):
                stocks_prices.drop(ticker, axis=1, inplace=True)

    stocks_prices.ffill(inplace=True)
    pickle_dump(obj=stocks_prices, filename='cleaned_nasdaq_dataframe')
    return stocks_prices

# machine learning algorithms


def xtrain_ytrain(adj_close_stocks_prices):
    """
    Splits the DataFrame into training and testing sets, normalizes the data, and prepares it for LSTM model training.

    Parameters:
        adj_close_stocks_prices (pandas.DataFrame): DataFrame containing adjusted close prices.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]: A tuple containing training and testing data along with the scaler.
    """
    split_index = int((len(adj_close_stocks_prices)) * 0.80)
    train_set = pd.DataFrame(adj_close_stocks_prices.iloc[0:split_index])
    test_set = pd.DataFrame(adj_close_stocks_prices.iloc[split_index:])

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

# correlation study


def plot_corr_matrix(dataframe):
    """
    Plot the correlation matrix heatmap for a given DataFrame.

    Parameters:
    dataframe (pandas.DataFrame): The DataFrame containing the correlation matrix to be visualized.

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
    seaborn.heatmap(dataframe, annot=True, cmap=cmap)
    plt.show()


def get_correlated_stocks(stocks_prices, tickers):
    """
    Get correlated stocks based on the correlation matrix of their prices within a given time period.

    Parameters:
    stocks_prices (pandas.DataFrame): DataFrame containing the prices of different stocks.
    tickers (list): List of ticker tickers representing the stocks.
    start_datetime (str): Start date and time in 'YYYY-MM-DD HH:MM:SS' format.
    end_datetime (str): End date and time in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing a numpy array of correlation coefficients
                                    and a numpy array of p-values.
    """
    # Make array that will store the ticker pairs, and their p-value
    corr_values = np.zeros([len(tickers), len(tickers)])
    pvalue_array = np.zeros([len(tickers), len(tickers)])
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            # Loop on matrix, get pair (i,j)
            vals_i = stocks_prices[tickers[i]].values
            vals_j = stocks_prices[tickers[j]].values
            # Get correlation coef and corresponding p-value of each pair
            r_ij, p_ij = ss.stats.pearsonr(vals_i, vals_j)
            corr_values[i, j] = r_ij
            pvalue_array[i, j] = p_ij
    pickle_dump(corr_values, 'correlationvalues_array')
    pickle_dump(pvalue_array, 'pvalues_array')
    
    return corr_values, pvalue_array

def corr_stocks_pair(corr_values, pvalue_array, tickers):
    """
    Create a DataFrame containing prices of correlated stocks and save it to a pickle file.

    Parameters:
    corr_values (np.ndarray): Array containing correlation coefficients.
    pvalue_array (np.ndarray): Array containing p-values.
    tickers (list): List of all ticker tickers in the dataset.

    Returns:
    list: A list containing tickers corresponding to the pair with the maximum correlation coefficient.
    """
    #filter based on test significance
    corr_values_filtered = np.where(pvalue_array > 0.05, corr_values, np.nan)
    # Get min and max correlation values
    min_corr = np.nanmin(corr_values_filtered)
    tmp_arr = corr_values_filtered.copy()
    # Make diagonal 0 (correlation with itself is 1, make 0)
    for i in range(len(tmp_arr)):
        tmp_arr[i, i] = 0
    # Calculate max correlation
    max_corr = np.nanmax(tmp_arr)
    # Get pair that has the max correlation value
    max_indexes = np.where(corr_values == max_corr)
    # Store pair as a tuple
    max_pair = [tickers[max_indexes[0][0]], tickers[max_indexes[0][1]]]

    corr_order = np.argsort(tmp_arr.flatten())
    corr_num = corr_order[-1]
    max_pair = [tickers[corr_num // len(tickers)], tickers[corr_num % len(tickers)]]
    
    pickle_dump(max_pair, 'df_maxcorr_pair')
    
    return max_pair
