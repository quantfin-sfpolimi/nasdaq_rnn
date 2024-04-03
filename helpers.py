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

# file saving with pickling


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

def load_dataframe(years, filename):
    """
    Loads stock price data either from a pickled file or downloads it online using the yfinance library.
    Returns a pandas DataFrame containing the stock prices and a list of tickers.

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
    if os.path.isfile("stocks_prices_dataframe.pkl"):
        stock_prices = pickle_load("stocks_prices_dataframe.pkl")
        tickers = stock_prices.columns.tolist()
    else:
        tickers = get_stockex_tickers(link=link)
        stock_prices = loaded_df(years=years, tickers=tickers, interval=interval)

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
        prices = obb.equity.price.historical(ticker ,start_date = start_date, end_date=end_date, provider="yfinance", interval=interval).to_df()
        stocks_dict[ticker] = prices['close']

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


def clean_df(percentage, tickers, stocks_prices):
    """
    Cleans the DataFrame by dropping stocks with NaN values exceeding the given percentage threshold.
    The cleaned DataFrame is pickled after the operation.

    Parameters:
    percentage : float
        Percentage threshold for NaN values. If greater than 1, it's interpreted as a percentage (e.g., 5 for 5%).
    tickers : List[str]
        List of ticker symbols.
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

    stocks_prices = final_clean_df(stocks_prices)
    pickle_dump(obj=stocks_prices, filename='cleaned_nasdaq_dataframe')
    return stocks_prices


def final_clean_df(adj_close_df):
    """
    Perform linear interpolation to fill missing values (NaNs) in the given DataFrame.

    Parameters:
    - adj_close_df (pandas.DataFrame): A DataFrame where rows represent tickers
      and columns represent dates. Each cell contains the adjusted closing price
      of the corresponding ticker on the respective date.

    Returns:
    pandas.DataFrame: A DataFrame with missing values filled using linear interpolation.
    If there were no missing values to interpolate, the original DataFrame is returned.
    """
    for day_index in adj_close_df.index:
        for ticker in adj_close_df.columns:
            try:
                value = adj_close_df.at[day_index, ticker]
                if pd.isna(value):
                    index_loc = adj_close_df.index.get_loc(day_index)
                    previous = adj_close_df.at[adj_close_df.index[index_loc - 1], ticker] if index_loc > 0 else None
                    later = adj_close_df.at[adj_close_df.index[index_loc + 1], ticker] if index_loc < len(adj_close_df) - 1 else None
                    if previous is not None and later is not None:
                        interpolated = (previous + later) / 2
                        adj_close_df.at[day_index, ticker] = interpolated
            except Exception as e:
                print(f"Error occurred: {e}")
    return adj_close_df

# machine learning algorithms

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
