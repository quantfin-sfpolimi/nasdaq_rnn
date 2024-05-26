
from helpermodules.df_cleaning import DataFrameHelper
from helpermodules.correlation_study import CorrelationAnalysis

df_nasdaq = DataFrameHelper(filename='nasdaq_dataframe', link='https://en.wikipedia.org/wiki/Nasdaq-100',months=2, frequency='1min')

#load data, clean data frame (closing stock prices)
df_nasdaq.getdata() 


#if over 10% of data is Nan, drop the ticker; remaining NAN will be replaced with (t-1)
df_nasdaq.clean_df(5)
print(df_nasdaq)
corr_study = CorrelationAnalysis(df_nasdaq)
corr_study.get_correlated_stocks()
corr_study.corr_stocks_pair()
corr_study.plot_corr_matrix()

