import io
import random
import string
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from plotly import graph_objs as go
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.plot import plot_plotly
from yahooquery import Ticker

sys.path.append('recommend')

import recommendations as rec

diagnostics_run = False
# Page selection options
page_tabs = ["**Forecast**", "**Diagnostics**", "**Recommendation**", "**Latest News**"]
indices = ['Nifty-500', 'Nasdaq-100', 'AEX']
st.set_page_config(page_title="Stock Forecasting App", layout="wide")
st.markdown('<h1 style="text-align: center;">Stock Forecast App</h1>', unsafe_allow_html=True)


class ForecastStockPrice:

    def __init__(self):
        self.csv_file_path = 'data/ind_nifty500list.csv'
        self.date_string = "2015-01-01"
        self.date_format = "%Y-%m-%d"
        self.START = datetime(2015, 1, 1).date()
        self.TODAY = date.today()

    @st.cache_data
    def get_stocks_from_csv_data(_self):
        """
        returns a dictionary with stock_name:stock_symbol and a list with industry name
        :return:
        """
        df = pd.read_csv(_self.csv_file_path, sep=',')
        stocks = dict(zip(df['Company Name'], df['Symbol'].astype(str) + '.NS'))
        return stocks

    def return_ticker_and_data(self, stocks):
        """

        :param stocks:
        :return:
        """
        stock_name = tuple(stocks.keys())
        selected_stock = st.sidebar.selectbox("**Select stock**", stock_name)
        ticker = stocks[selected_stock]
        start_date = st.sidebar.date_input("**Start Date**", self.START)
        end_date = st.sidebar.date_input("**End Date**", self.TODAY)
        return ticker, start_date, end_date

    @staticmethod
    @st.cache_data
    def fetch_stock_news(ticker):
        """

        :return:
        """
        r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json")
        data = r.json()
        ticker_symbol_name = ticker.split(".")
        st.subheader(f"Latest news for {ticker_symbol_name[0]}")
        for message in data['messages']:
            st.image(message['user']['avatar_url'])
            st.write(message['user']['username'])
            st.write(message['created_at'])
            st.write(message['body'])
            st.write("---")

    @staticmethod
    @st.cache_resource
    def show_ticker_data(ticker, start_date, end_date):
        """

        :param ticker:
        :param start_date:
        :param end_date:
        :return:
        """
        tickerData = yf.Ticker(ticker)
        historical_data = tickerData.history(start=start_date, end=end_date)
        historical_data = historical_data.reset_index()
        historical_data['Date'] = historical_data['Date'].dt.tz_localize(None)

        # st.write(tickerData.info)
        string_name = tickerData.info['longName']
        st.header(f'**{string_name}**')

        string_rec = tickerData.info['recommendationKey']
        st.info(f"Recommendation: {string_rec}")

        string_summary = tickerData.info['longBusinessSummary']
        st.info(string_summary)

        # Ticker Data
        st.subheader('Ticker data')
        historical_data = historical_data.rename_axis('S.No.')
        st.dataframe(historical_data.tail().style.set_properties(**{'text-align': 'center'}), use_container_width=True)
        return historical_data

    @staticmethod
    @st.cache_data
    def plot_raw_data(historical_data):
        """

        :return:
        """
        # Create the candlestick chart figure
        fig = go.Figure(data=go.Candlestick(x=historical_data['Date'],
                                            open=historical_data['Open'],
                                            high=historical_data['High'],
                                            low=historical_data['Low'],
                                            close=historical_data['Close']))
        # Customize the layout
        fig.update_layout(
            title="Price Action Graph",
            xaxis_title="Date",
            yaxis_title="Price"
        )
        # Render the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def outlier_data():
        lockdowns = pd.DataFrame([
            {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
            {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
            {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
            {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
            {'holiday': 'war_start', 'ds': '2022-02-23', 'lower_window': 0, 'ds_upper': '2022-03-09'},
        ])

        lockdowns[['ds', 'ds_upper']] = lockdowns[['ds', 'ds_upper']].apply(pd.to_datetime)
        lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
        return lockdowns

    @staticmethod
    def handle_outliers(update_outliers, lockdowns):
        if update_outliers == 'Yes':
            add_outliers = st.selectbox("**Choose ADD or REMOVE**", ('Remove', 'Add'), key='outliers')

            if add_outliers == 'Add':
                start_date1 = st.text_input("Start Date")
                end_date1 = st.text_input("End Date")
                outlier_name = st.text_input("Outlier Name")

                if st.button('Add'):
                    my_dict = {'holiday': outlier_name, 'ds': start_date1, 'ds_upper': end_date1}
                    new_row = pd.DataFrame([my_dict])
                    lockdowns = pd.concat([lockdowns, new_row], ignore_index=True)
                    lockdowns[['ds', 'ds_upper']] = lockdowns[['ds', 'ds_upper']].apply(pd.to_datetime)
                    lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
                    st.success("Data added successfully!")

            elif add_outliers == 'Remove':
                options1 = list(range(0, len(lockdowns)))
                row_num = st.selectbox("Which row you want to delete?", options1, index=options1[-1])

                if st.button('Remove'):
                    lockdowns = lockdowns.drop(row_num)
                    st.success("Data removed successfully")

            st.write("**Current Outliers Data**")
            st.table(lockdowns)
        else:
            pass

        return lockdowns

    @staticmethod
    def get_data_to_predict_future(historical_data):
        lockdowns = ForecastStockPrice.outlier_data()
        st.subheader("Do you want to update outlier data?")
        characters = string.ascii_letters + string.digits + string.punctuation
        random_string = ''.join(random.choice(characters) for _ in range(10))
        update_outliers = st.selectbox("**Do you want to update Outliers**", ('No', 'Yes'), key=random_string)
        if update_outliers == 'Yes':
            with st.expander("Expand Outlier Data"):
                lockdowns = ForecastStockPrice.handle_outliers(update_outliers, lockdowns)

        st.subheader("Forecast Stock Price")
        df_train = historical_data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

        st.subheader("Model Forecast Parameters")
        col01, col11, col12 = st.columns(3)
        n_days = col01.slider("**Days of prediction**", 1, 365, 180)
        changepoint_prior_scale = col11.slider("**Changepoint Prior Scale**", 0.05, 0.5, 0.2)
        changepoint_range = col12.slider("**Changepoint Range**", 0.8, 0.95, 0.95)
        return changepoint_range, changepoint_prior_scale, lockdowns, df_train, n_days

    @staticmethod
    @st.cache_data
    def predict_future(changepoint_range, changepoint_prior_scale, lockdowns, df_train, n_days):
        """

        :param changepoint_range:
        :param changepoint_prior_scale:
        :param lockdowns:
        :param df_train:
        :param n_days:
        :return:
        """
        with st.spinner("Future Share Price prediction in progress..."):
            stock_mod = Prophet(changepoint_range=changepoint_range,
                                changepoint_prior_scale=changepoint_prior_scale,
                                holidays=lockdowns)
            stock_mod.fit(df_train)
            future = stock_mod.make_future_dataframe(periods=n_days)
            forecast = stock_mod.predict(future)
            return forecast, stock_mod

    @staticmethod
    @st.cache_data
    def plot_forcast_data(forecast_var, _stock_mod):
        st.subheader('Forecast data')
        forecast_var.index.name = 'S.No.'
        with st.spinner("Forcast data in progress..."):
            st.dataframe(
                forecast_var[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower',
                              'trend_upper']].tail().style.set_properties(**{'text-align': 'center'}),
                use_container_width=True)
            st.success("Forcast data is fetched successfully!")

        st.subheader('Forecast graph')
        with st.spinner("Plotting forcast graph..."):
            fig1 = plot_plotly(_stock_mod, forecast_var, changepoints=True)
            st.plotly_chart(fig1, use_container_width=True)
            st.success("Forcast data is plotted successfully!")

        st.subheader("Forecast components")
        with st.spinner("Plotting forecast components..."):
            fig2 = _stock_mod.plot_components(forecast_var)
            st.write(fig2)
            st.success("Forcast components are fetched successfully!")

    def run_forecast(self):
        st.title('Forecast')
        stocks = self.get_stocks_from_csv_data()
        ticker, start_date, end_date = self.return_ticker_and_data(stocks)
        historical = ForecastStockPrice.show_ticker_data(ticker, start_date, end_date)
        ForecastStockPrice.plot_raw_data(historical)
        changepoint_range, changepoint_prior_scale, lockdowns, df_train, n_days = ForecastStockPrice.get_data_to_predict_future(
            historical)
        forecast, stock_mod = ForecastStockPrice.predict_future(changepoint_range, changepoint_prior_scale,
                                                                lockdowns,
                                                                df_train, n_days)
        ForecastStockPrice.plot_forcast_data(forecast, stock_mod)
        return ticker, stock_mod

    @staticmethod
    @st.cache_data
    def run_diagnostics(initial_training_period, change_period, forecast_horizon, _stock_mod):
        """

        :param initial_training_period:
        :param change_period:
        :param forecast_horizon:
        :param _stock_mod:
        :return:
        """
        st.write("Cross Validation")
        df_cv = cross_validation(_stock_mod, initial=f"{initial_training_period} days",
                                 period=f"{change_period} days",
                                 horizon=f"{forecast_horizon} days")
        df_cv.index.name = 'S.No.'
        st.dataframe(df_cv.head().style.set_properties(**{'text-align': 'center'}), use_container_width=True)

        st.write("Performance Metrics")
        df_p = performance_metrics(df_cv)
        df_p['horizon'] = df_p['horizon'].astype(str)
        df_p.index.name = 'S.No'
        st.dataframe(df_p.head().style.set_properties(**{'text-align': 'center'}), use_container_width=True)

        st.write("Plot Cross Validation Metrics")
        fig3 = plot_cross_validation_metric(df_cv, metric='mape')
        st.write(fig3)

    def diagnostics_data(self, stock_mod):
        st.subheader("Do You want to run Diagnostics?")
        diagnostics_option = st.selectbox("**Choose an option**", ('Yes', 'No'), key='diagnostics')

        if diagnostics_option == 'Yes':
            st.subheader("Diagnostics parameters")
            col1, col2, col3 = st.columns(3)
            initial_training_period = col1.slider("**Initial Training Period**", 100, 2500, 500)
            change_period = col2.slider("**Spacing Between Cutoff Dates**", 25, 100, 50)
            forecast_horizon = col3.slider("**Forecast Horizon**", 7, 365, 180)

            if st.button('Run Diagnostics'):
                with st.spinner("Backtesting is in progress..."):
                    self.run_diagnostics(initial_training_period, change_period, forecast_horizon, stock_mod)
        else:
            st.write("No Diagnostics to run.")

    @staticmethod
    @st.cache_data
    def get_recommendations(stocks):
        with ThreadPoolExecutor() as executor:
            recommendations = list(
                executor.map(lambda symbol: Ticker(symbol).financial_data[symbol]['recommendationKey'],
                             stocks.values()))
        return recommendations

    def show_recommendations(self):
        stocks = self.get_stocks_from_csv_data()
        with st.spinner("Fetching recommendations..."):
            recommendations = self.get_recommendations(stocks)
        company_names = list(stocks.keys())[:len(recommendations)]
        df = pd.DataFrame({
            'Company Name': company_names,
            'Recommendation': recommendations
        })
        df = df.rename_axis('S.No.')
        st.dataframe(df.style.set_properties(**{'text-align': 'center'}), use_container_width=True)
        if st.button("Download as Excel"):
            if len(df) > 0:
                output = io.BytesIO()
                excel_writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df.to_excel(excel_writer, index=False, sheet_name='Recommendations')
                excel_writer.close()
                output.seek(0)
                excel_data = output.getvalue()
                st.download_button(
                    label="Download Recommendations",
                    data=excel_data,
                    file_name="recommend.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No data available to download.")

    def run_recommendations(self):
        st.title("Recommendation")
        options = ['No', 'Yes']
        index = st.selectbox("Provide Index Name", indices)
        obj = rec.Recommender(index)
        db_update = st.selectbox("Do you want to Update Database?", options)
        if db_update == 'Yes':
            with st.spinner("DB update in in progress..."):
                obj.update_db()
                st.success("Database updated successfully!")
        elif db_update == 'No':
            if st.button("Stock Recommendations Based On Indicators"):
                st.subheader("Stock Recommendations")
                with st.spinner(f"Fetching recommendation for {index}..."):
                    signals = obj.recommender()
                    df = pd.DataFrame({'Signals': signals})
                    df = df.rename_axis("S.No")
                    st.dataframe(df.style.set_properties(**{'text-align': 'center'}), use_container_width=True)
        if st.button("Recommendations Based On Analysts Recommendations"):
            st.write("**General Recommendations**")
            self.show_recommendations()

    def run_app(self):
        tab1, tab2, tab3, tab4 = st.tabs(page_tabs)
        with tab1:
            ticker, stock_mod = self.run_forecast()
        with tab2:
            self.diagnostics_data(stock_mod)
        with tab3:
            self.run_recommendations()
        with tab4:
            ForecastStockPrice.fetch_stock_news(f"{ticker}E")


if __name__ == "__main__":
    app = ForecastStockPrice()
    app.run_app()
