
from df_cleaning import DataFrameHelper 
from correlation_study import CorrelationAnalysis
from memory_handling import PickleHelper
from dotenv import load_dotenv
import pandas as pd

df_nasdaq = DataFrameHelper(filename='cleaned_nasdaq_dataframe', link='https://en.wikipedia.org/wiki/Nasdaq-100', months=1, frequency='1min')



#load data, clean data frame (closing stock prices)
df_nasdaq.load()

print(df_nasdaq.tickers)
print(df_nasdaq.dataframe)

corr_study = CorrelationAnalysis(df_nasdaq.dataframe, df_nasdaq.tickers, '2024-04-10 09:30:00', '2024-05-08 15:59:00')

corr_study.print_corr(0.965, 2)

#FIXME: temporary fix for clean_df
#df_nasdaq.dataframe.fillna(method='ffill', inplace=True)
#df_nasdaq.dataframe.dropna(inplace=True)

#PickleHelper(obj=df_nasdaq.dataframe).pickle_dump(filename='cleaned_nasdaq_dataframe')

#if over 10% of data is Nan, drop the ticker; remaining NAN will be replaced with (t-1)
#df_nasdaq.clean_df(5)

'''

shifted_dataframe = df_nasdaq.dataframe.copy(deep = True)
shifted_dataframe = shifted_dataframe.shift(10, axis = 0)

for ticker in df_nasdaq.tickers:
    shifted_dataframe.rename({ticker:ticker+'_shifted'},axis = 1, inplace = True)

concat_dataframe =  pd.concat([df_nasdaq.dataframe.iloc[10: , :], shifted_dataframe.iloc[10: , :]], axis=1)

print(concat_dataframe)


corr_study = CorrelationAnalysis(dataframe=concat_dataframe, tickers=list(concat_dataframe.columns), start_datetime='2024-04-10 09:30:00', end_datetime='2024-05-08 16:00:00')

corr_study.get_correlated_stocks()
corr_study.corr_stocks_pair()
corr_study.plot_corr_matrix()

'''



