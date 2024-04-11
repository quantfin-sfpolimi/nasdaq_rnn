from openbb import obb
import pandas as pd
from helpers import load_dataframe, clean_stocks_prices, get_correlated_stocks, plot_corr_matrix, corr_stocks_pair

def main():
    # simo's login with obb platform credetial
    obb.account.login(email='simo05062003@gmail.com', password='##2yTFb2F4Zd9z')
    #load data, clean data frame (closing stock prices)
    stocks_prices, tickers = load_dataframe(years=10, filename='cleaned_nasdaq_dataframe', link='https://en.wikipedia.org/wiki/Nasdaq-100', interval='5m') 
    #if over 10% of data is Nan, drop the ticker; remaining NAN will be replaced with (t-1)
    stocks_prices = clean_stocks_prices(10, tickers=tickers, stocks_prices=stocks_prices)
    stocks_prices.info()
    corr_values, pvalues = get_correlated_stocks(stocks_prices=stocks_prices, tickers=tickers)

    winner_pair = corr_stocks_pair(corr_values=corr_values, pvalue_array=pvalues, tickers=tickers)
    plot_corr_matrix(winner_pair)

if __name__ == '__main__':
    main()