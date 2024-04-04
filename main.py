from openbb import obb
import pandas as pd
from helpers import load_dataframe, clean_df

# simo's login with obb platform credetial
obb.account.login(email='simo05062003@gmail.com', password='##2yTFb2F4Zd9z')
#load data, clean data frame (closing stock prices)
stocks_prices, tickers =    load_dataframe(years=10, filename='cleaned_nasdaq_dataframe', link='https://en.wikipedia.org/wiki/Nasdaq-100', interval='5m') 
#if over 10% of data is Nan, drop the ticker; remaining NAN will be replaced with average of (t-1) and (t+1)
clean_df(10, tickers=tickers, stocks_prices=stocks_prices)
print(stocks_prices)




