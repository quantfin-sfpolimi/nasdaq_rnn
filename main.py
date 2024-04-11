from openbb import obb
import pandas as pd
from helpers import DataFrameHelper, CorrelationAnalysis

# simo's login with obb platform credetial
obb.account.login(email='simo05062003@gmail.com', password='##2yTFb2F4Zd9z')
#load data, clean data frame (closing stock prices)
stocks_prices, tickers = DataFrameHelper.load_dataframe(years=10, filename='cleaned_nasdaq_dataframe', link='https://en.wikipedia.org/wiki/Nasdaq-100', interval='5m') 
#if over 10% of data is Nan, drop the ticker; remaining NAN will be replaced with (t-1)
stocks_prices = DataFrameHelper.clean_df(10, tickers=tickers, stocks_prices=stocks_prices)

corr_stocks_dict, corr_stocks_list = CorrelationAnalysis.get_correlated_stocks(stocks_prices, tickers, '2024-03-01 09:30:00', '2024-03-31 15:30:00')

corr_stocks_df = CorrelationAnalysis.corr_df(corr_stocks_dict, corr_stocks_list, tickers=tickers, stocks_prices=stocks_prices)

CorrelationAnalysis.plot_corr_matrix(corr_stocks_df)
print("GO")
