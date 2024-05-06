
from helpers import DataFrameHelper, CorrelationAnalysis
from dotenv import load_dotenv

df_nasdaq = DataFrameHelper(filename='cleaned_nasdaq_dataframe', link='https://en.wikipedia.org/wiki/Nasdaq-100', years=1, interval='1min')



#load data, clean data frame (closing stock prices)
df_nasdaq.load()


#if over 10% of data is Nan, drop the ticker; remaining NAN will be replaced with (t-1)
#df_nasdaq.clean_df(5)

#FIXME: temporary fix for clean_df
df_nasdaq.dataframe.fillna(method='ffill', inplace=True)
df_nasdaq.dataframe.dropna(inplace=True)


corr_study = CorrelationAnalysis(dataframe=df_nasdaq.dataframe, tickers=df_nasdaq.tickers, start_datetime='2024-04-17 09:30:00', end_datetime='2024-05-03 15:59:00')

corr_study.get_correlated_stocks()
corr_study.corr_stocks_pair()
corr_study.plot_corr_matrix()


