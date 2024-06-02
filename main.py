
from helpermodules.df_dataretrieval import IndexData_Retrieval
from helpermodules.correlation_study import CorrelationAnalysis

df_nasdaq = IndexData_Retrieval(filename='nasdaq_dataframe', link='https://en.wikipedia.org/wiki/Nasdaq-100',months=2, frequency='1min')

#load data, clean data frame (closing stock prices)
df_nasdaq.getdata() 


#FIXME: if over 10% of data is Nan, drop the ticker; remaining NAN will be replaced with (t-1)
#df_nasdaq.clean_df(5)
#TODO: readapt correlation analysis to new format df
corr_study = CorrelationAnalysis(df_nasdaq.df, df_nasdaq.tickers)
corr_study.print_cross_corr(0.965, 5)
corr_study.print_corr(0.965, 5)

