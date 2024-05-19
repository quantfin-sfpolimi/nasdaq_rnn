import pandas as pd
from df_cleaning import DataFrameHelper
import matplotlib.pyplot as plt

class BacktestingPairs:
    def __init__(self, df_helper, money, leading_stock, lagging_stock):
        self.money = money
        df_helper.load()
        self.leading_stock = df_helper.dataframe[leading_stock]
        self.lagging_stock = df_helper.dataframe[lagging_stock]
        self.money_history = []

    def strategy_1(self, lag, start_date, end_date):

        self.leading_stock = self.leading_stock.loc[start_date:end_date]
        self.lagging_stock = self.lagging_stock.loc[start_date:end_date]
        
        for i in range(0, len(self.leading_stock)-lag-1):
            if self.leading_stock.iloc[i] < self.leading_stock.iloc[i+1]:
                delta = self.lagging_stock.iloc[i+lag+1] - self.lagging_stock.iloc[i+lag]
                self.money += delta
                self.money_history.append(delta)

        print(self.money)
        plt.plot(self.money_history)
        plt.show()


