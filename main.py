
from helpers import DataFrameHelper, CorrelationAnalysis

df_nasdaq = DataFrameHelper(filename='cleaned_nasdaq_dataframe', link='https://en.wikipedia.org/wiki/Nasdaq-100', years=10, interval='2m')

#load data, clean data frame (closing stock prices)
df_nasdaq.load()

#if over 10% of data is Nan, drop the ticker; remaining NAN will be replaced with (t-1)
df_nasdaq.clean_df(10)

corr_study = CorrelationAnalysis(dataframe=df_nasdaq.dataframe, tickers=df_nasdaq.tickers, start_datetime='2024-03-01 09:30:00', end_datetime='2024-03-31 15:30:00')

corr_study.get_correlated_stocks()
corr_study.corr_stocks_pair()
corr_study.plot_corr_matrix()

