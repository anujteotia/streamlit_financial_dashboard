import sqlalchemy
import pymysql
import pandas as pd
import requests
import yfinance as yf


pymysql.install_as_MySQLdb()
indices = ['Nifty-500', 'Nasdaq-100', 'AEX']


def schema_creator(index1):
    engine = sqlalchemy.create_engine('mysql://root:123456789@localhost:3306/')
    with engine.connect() as conn:
        conn.execute(sqlalchemy.schema.CreateSchema(index1))


for index in indices:
    schema_creator(index)

nifty500 = pd.read_csv("data/ind_nifty500list.csv")
nifty500 = nifty500.Symbol.to_list()
nifty500 = [i + '.NS' for i in nifty500]

aex_index_wiki = "https://en.wikipedia.org/wiki/AEX_index"
response = requests.get(aex_index_wiki)
table = pd.read_html(response.text)[3]
aex_index = table['Ticker symbol'].tolist()
aex_index = [i + '.AS' for i in aex_index]

nasdaq_100_wiki = "https://en.wikipedia.org/wiki/NASDAQ-100"
response = requests.get(nasdaq_100_wiki)
tickers_table = pd.read_html(response.text)[4]
nasdaq_100 = tickers_table["Ticker"].tolist()


mapper = {'Nifty-500': nifty500, 'AEX': aex_index, 'Nasdaq-100': nasdaq_100}

for index in indices:
    engine1 = sqlalchemy.create_engine('mysql://root:123456789@localhost:3306/'+index)
    for symbol in mapper[index]:
        df = yf.download(symbol, start='2020-01-01')
        df = df.reset_index()
        df.to_sql(symbol, engine1)

