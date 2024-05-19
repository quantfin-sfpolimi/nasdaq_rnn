
from df_cleaning import DataFrameHelper 
from correlation_study import CorrelationAnalysis
from memory_handling import PickleHelper
from backtesting import BacktestingPairs 
from dotenv import load_dotenv
import pandas as pd

df_nasdaq = DataFrameHelper(filename='cleaned_nasdaq_dataframe', link='https://en.wikipedia.org/wiki/Nasdaq-100', months=1, frequency='1min')
df_nasdaq.load()

print(df_nasdaq.dataframe)

#corr_study = CorrelationAnalysis(df_nasdaq.dataframe, df_nasdaq.tickers, '2024-04-07 09:30:00', '2024-04-21 15:59:00')

#corr_study.print_corr(0.965, 5)


backtest = BacktestingPairs(df_nasdaq, 10000, 'KDP', 'AZN')
backtest.strategy_1(2, '2024-04-22 09:30:00', '2024-05-08 15:59:00')



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



