import pandas as pd
import sqlalchemy
import pymysql
import ta
import numpy as np
import yfinance as yf
import streamlit as st

pymysql.install_as_MySQLdb()


class Recommender:
    engine = sqlalchemy.create_engine('mysql://root:123456789@localhost:3306')

    def __init__(self, index):
        self.index = index

    def get_tables(self):
        query = f"""SELECT table_name FROM information_schema.tables WHERE table_schema = '{self.index}'"""
        df = pd.read_sql(query, self.engine)
        df['Schema'] = self.index
        return df

    def max_date(self):
        req = f'`{self.index}`' + '.' + f'`{self.get_tables().TABLE_NAME[0]}`'
        return pd.read_sql(f"SELECT MAX(Date) FROM {req}", self.engine)

    def update_db(self):
        max_date = self.max_date()['MAX(Date)'][0]
        engine = sqlalchemy.create_engine('mysql://root:123456789@localhost:3306/' + self.index)
        for symbol in self.get_tables().TABLE_NAME:
            data =yf.download(symbol, start=max_date)
            data = data[data.index > max_date]
            data = data.reset_index()
            data.to_sql(symbol, engine, if_exists='append')
        print(f"{self.index} successfully updated")

    def get_prices(self):
        prices = []
        for table, schema in zip(self.get_tables().TABLE_NAME, self.get_tables().Schema):
            sql = f'`{schema}`'+'.'+f'`{table}`'
            prices.append(pd.read_sql(f"SELECT Date, Close FROM {sql}", self.engine))
        return prices

    @staticmethod
    def macd_decision(df):
        df['MACD_diff'] = ta.trend.macd_diff(df.Close)
        df['Decision MACD'] = np.where((df.MACD_diff > 0) & (df.MACD_diff.shift(1) < 0), True, False)

    @staticmethod
    def golden_cross_decision(df):
        df['SMA20'] = ta.trend.sma_indicator(df.Close, window=20)
        df['SMA50'] = ta.trend.sma_indicator(df.Close, window=50)
        df['Signal'] = np.where(df['SMA20'] > df['SMA50'], True, False)
        df['Decision GC'] = df.Signal.diff()

    def rsi_sma_decision(self, df):
        df['RSI'] = ta.momentum.rsi(df.Close, window=10)
        df['SMA200'] = ta.trend.sma_indicator(df.Close, window=200)
        df['Decision RSI/SMA'] = np.where((df.Close > df.SMA200) & (df.RSI < 30), True, False)

    @st.cache_data
    def apply_technicals(self):
        prices = self.get_prices()
        for frame in prices:
            Recommender.macd_decision(frame)
            Recommender.golden_cross_decision(frame)
            self.rsi_sma_decision(frame)
        return prices

    @st.cache_data
    def recommender(self):
        signals = []
        indicators = ['Decision MACD', 'Decision GC', 'Decision RSI/SMA']
        for symbol, frame in zip(self.get_tables().TABLE_NAME, self.apply_technicals()):
            if frame.empty is False:
                for indicator in indicators:
                    if frame[indicator].iloc[-1]:
                        signals.append(f"{indicator} Buying signal for {symbol}")
        return signals


if __name__ == "__main__":
    nifty_instance = Recommender('Nifty-500')
    nifty_instance.update_db()
    print(nifty_instance.recommender())
