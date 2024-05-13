
import datetime as dt
import os
import pandas as pd
from twelvedata import TDClient
import re
from memory_handling import PickleHelper
from dotenv import load_dotenv
import time

class DataFrameHelper:
    """
    A class for downloading and processing historical stock price data using the Twelve Data API.

    Parameters:
        filename (str): Name of the pickle file to save or load DataFrame.
        link (str): URL link to a Wikipedia page containing stock exchange information.
        interval (str): Time frequency of historical data to load (e.g., '1min', '1day', '1W').
        frequency (str): Frequency of data intervals ('daily', 'weekly', 'monthly', etc.).
        years (int, optional): Number of years of historical data to load (default: None).
        months (int, optional): Number of months of historical data to load (default: None).

    Methods:
        load():
            Loads a DataFrame of stock price data from a pickle file if it exists, otherwise creates a new DataFrame.
            Returns:
                pandas.DataFrame or None: DataFrame containing stock price data if loaded successfully, otherwise None.

        get_stockex_tickers():
            Retrieves ticker symbols from a Wikipedia page containing stock exchange information.
            Returns:
                List[str]: List of ticker symbols extracted from the specified Wikipedia page.

        loaded_df():
            Downloads historical stock price data for the specified time window and tickers using the Twelve Data API.
            Returns:
                pandas.DataFrame or None: DataFrame containing downloaded stock price data if successful, otherwise None.
    """

    def __init__(self, filename, link, frequency, years=None, months=None):
        self.filename = filename
        self.link = link
        self.frequency = frequency
        self.tickers = []
        self.years = years
        self.months = months

    def load(self):
        """
        Load a DataFrame of stock price data from a pickle file if it exists, otherwise create a new DataFrame.
        Returns:
            pandas.DataFrame or None: DataFrame containing stock price data if loaded successfully, otherwise None.
        """
        if not re.search("^.*\.pkl$", self.filename):
            self.filename += ".pkl"
        file_path = "./pickle_files/" + self.filename

        if os.path.isfile(file_path):
            self.dataframe = PickleHelper.pickle_load(self.filename).obj
            self.tickers = self.dataframe.columns.tolist()
            return self.dataframe
        else:
            self.tickers = self.get_stockex_tickers()
            self.dataframe = self.loaded_df()

        return None


    def get_stockex_tickers(self):
        """
        Retrieves ticker symbols from a Wikipedia page containing stock exchange information.
        Returns:
            List[str]: List of ticker symbols extracted from the specified Wikipedia page.
        """
        tables = pd.read_html(self.link)
        df = tables[4]
        df.drop(['Company', 'GICS Sector', 'GICS Sub-Industry'],
                axis=1, inplace=True)
        tickers = df['Ticker'].values.tolist()
        return tickers
    
    def concat_data(self, td, start_date, output_size, ticker, interval):

        data_to_load_tmp = output_size
        tmp_data = start_date
        parts = []

        for i in range(1, (output_size // 5000)+1):
            if (data_to_load_tmp >= 5000):
                ts = td.time_series(
                    symbol=ticker,
                    interval=interval,
                    outputsize=5000,
                    end_date = tmp_data,
                    timezone="America/New_York",
                ).as_pandas()
                data_to_load_tmp -= 5000
                parts.append(ts)
                tmp_data = parts[i-1].index.tolist()[-1] - dt.timedelta(minutes=1)
            if (data_to_load_tmp < 5000):
                ts = td.time_series(
                    symbol=ticker,
                    interval=interval,
                    outputsize=data_to_load_tmp,
                    end_date = tmp_data,
                    timezone="America/New_York",
                ).as_pandas()
                parts.append(ts)
        print(pd.concat(parts))
        return pd.concat(parts)

    def loaded_df(self):
        """
        Downloads historical stock price data for the specified time window and tickers using the Twelve Data API.
        Returns:
            pandas.DataFrame or None: DataFrame containing downloaded stock price data if successful, otherwise None.
        """
        
        if self.years is not None and self.months is None:
            time_window_months = self.years * 12
        elif self.months is not None and self.years is None:
            time_window_months = self.months
        else:
            raise ValueError("Exactly one of 'years' or 'months' should be provided.")

        '''
        end_date = dt.date.today()
        start_date = end_date - pd.DateOffset(months=time_window_months)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        '''

        load_dotenv()
        API_KEY = os.getenv('API_KEY')
        td = TDClient(apikey=API_KEY)

        start_date = dt.datetime.now().replace(hour=16, minute=0 , second=0 ,microsecond=0) + dt.timedelta(days = -2)
        output_size = time_window_months*8190

        stocks_dict = {}

        for i, ticker in enumerate(self.tickers):
            print('Getting {} ({}/{})'.format(ticker, i, len(self.tickers)))
            dataframe = self.concat_data(td, start_date, output_size, ticker, self.frequency)  # 8190 trading minutes in a month
            stocks_dict[ticker] = dataframe['close']
            print(stocks_dict)

        return pd.DataFrame.from_dict(stocks_dict)

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

        self.dataframe.fillna(method='ffill', inplace=True)
        #FIXME: fml this doesn't work if i have consecutive days
        PickleHelper(obj=self.dataframe).pickle_dump(filename='cleaned_nasdaq_dataframe')

