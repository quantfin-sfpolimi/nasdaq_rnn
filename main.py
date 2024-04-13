from openbb import obb
import pandas as pd
from helpers import DataFrameHelper, CorrelationAnalysis

dataFrame_nasdaq = DataFrameHelper(filename='cleaned_nasdaq_dataframe', link='https://en.wikipedia.org/wiki/Nasdaq-100', years=10, interval='5m')

# simo's login with obb platform credetial
obb.account.login(email='simo05062003@gmail.com', password='##2yTFb2F4Zd9z')

#load data, clean data frame (closing stock prices)
dataFrame_nasdaq.load()

#if over 10% of data is Nan, drop the ticker; remaining NAN will be replaced with (t-1)
dataFrame_nasdaq.clean_df(10)

corr_analyses_nasdaq = CorrelationAnalysis(prices=dataFrame_nasdaq.prices, tickers=dataFrame_nasdaq.tickers, start_datetime='2024-03-01 09:30:00', end_datetime='2024-03-31 15:30:00')
corr_stocks_dict, corr_stocks_list = corr_analyses_nasdaq.get_correlated_stocks()

corr_analyses_nasdaq.df = corr_analyses_nasdaq.corr_df(corr_stocks_dict, corr_stocks_list)

corr_analyses_nasdaq.plot_corr_matrix()
