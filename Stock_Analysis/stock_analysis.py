import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from alpha_vantage.timeseries import TimeSeries

st.set_page_config(page_title="Stock Market Forecast Dashboard", layout="wide")
st.title("📈 Stock Market Forecast Dashboard")


st.sidebar.header("Settings")
key = "TZJ1U84UFAIIO0FW"  
symbol = st.sidebar.text_input("Stock Symbol", value="TSLA")
pred_steps = st.sidebar.slider("Prediction Horizon (days)", 7, 90, 30)


ts = TimeSeries(key, output_format='pandas')
data, meta = ts.get_weekly_adjusted(symbol=symbol)
data.columns = [col.split('. ')[1] for col in data.columns]
data.index.name = 'Date'
data = data.sort_index()

st.subheader(f"📊 Historical Stock Prices for {symbol}")
st.line_chart(data['close'])

def create_supervised(data, n_lags=60):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


results = {}
metrics = {}


tabs = st.tabs([
    "ARIMA", "SARIMA", "Prophet", "LSTM", "XGBoost", "LightGBM", "CatBoost", "Random Forest", "SVR", "Ridge"
])

# --- ARIMA ---
with tabs[0]:
    st.markdown("#### ARIMA Forecast")
    model_arima = ARIMA(data['close'], order=(5, 1, 0))
    results_arima = model_arima.fit()
    forecast_arima = results_arima.forecast(steps=pred_steps)
    st.line_chart(forecast_arima)
    results['ARIMA'] = forecast_arima.values
    metrics['ARIMA'] = mean_squared_error(data['close'][-pred_steps:], forecast_arima[:pred_steps]) if len(data) > pred_steps else np.nan

# --- SARIMA ---
with tabs[1]:
    st.markdown("#### SARIMA Forecast")
    model_sarima = SARIMAX(data['close'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    results_sarima = model_sarima.fit(disp=False)
    forecast_sarima = results_sarima.forecast(steps=pred_steps)
    st.line_chart(forecast_sarima)
    results['SARIMA'] = forecast_sarima.values
    metrics['SARIMA'] = mean_squared_error(data['close'][-pred_steps:], forecast_sarima[:pred_steps]) if len(data) > pred_steps else np.nan

# --- Prophet ---
with tabs[2]:
    st.markdown("#### Prophet Forecast")
    prophet_df = data.reset_index()[['Date', 'close']]
    prophet_df.columns = ['ds', 'y']
    model_prophet = Prophet(daily_seasonality=False, yearly_seasonality=True)
    model_prophet.fit(prophet_df)
    future = model_prophet.make_future_dataframe(periods=pred_steps)
    forecast_prophet = model_prophet.predict(future)
    st.line_chart(forecast_prophet[['ds', 'yhat']].set_index('ds')[-pred_steps:])
    results['Prophet'] = forecast_prophet['yhat'][-pred_steps:].values
    metrics['Prophet'] = mean_squared_error(prophet_df['y'][-pred_steps:], forecast_prophet['yhat'][-pred_steps:]) if len(prophet_df) > pred_steps else np.nan

# --- LSTM ---
with tabs[3]:
    st.markdown("#### LSTM Forecast")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['close']])
    X, y = create_supervised(data_scaled)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model_lstm.add(LSTM(50))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    model_lstm.fit(X, y, epochs=10, batch_size=32, verbose=0)
    pred_input = data_scaled[-60:]
    pred_input = pred_input.reshape(1, 60, 1)
    lstm_predictions = []
    for _ in range(pred_steps):
        next_pred = model_lstm.predict(pred_input, verbose=0)[0][0]
        lstm_predictions.append(next_pred)
        pred_input = np.append(pred_input[:, 1:, :], [[[next_pred]]], axis=1)
    lstm_predictions_actual = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=pred_steps)
    lstm_forecast_df = pd.DataFrame({'Date': future_dates, 'LSTM_Forecast': lstm_predictions_actual.flatten()})
    lstm_forecast_df.set_index('Date', inplace=True)
    st.line_chart(lstm_forecast_df)
    results['LSTM'] = lstm_predictions_actual.flatten()
    metrics['LSTM'] = mean_squared_error(data['close'][-pred_steps:], lstm_predictions_actual.flatten()) if len(data) > pred_steps else np.nan

# --- XGBoost ---
with tabs[4]:
    st.markdown("#### XGBoost Forecast")
    X, y = create_supervised(data_scaled)
    model_xgb = XGBRegressor(n_estimators=100)
    model_xgb.fit(X, y)
    pred_input = data_scaled[-60:]
    preds = []
    for _ in range(pred_steps):
        next_pred = model_xgb.predict(pred_input.reshape(1, -1))[0]
        preds.append(next_pred)
        pred_input = np.append(pred_input[1:], [[next_pred]], axis=0)
    preds_actual = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    st.line_chart(preds_actual)
    results['XGBoost'] = preds_actual
    metrics['XGBoost'] = mean_squared_error(data['close'][-pred_steps:], preds_actual) if len(data) > pred_steps else np.nan

# --- LightGBM ---
with tabs[5]:
    st.markdown("#### LightGBM Forecast")
    model_lgb = LGBMRegressor(n_estimators=100)
    model_lgb.fit(X, y)
    pred_input = data_scaled[-60:]
    preds = []
    for _ in range(pred_steps):
        next_pred = model_lgb.predict(pred_input.reshape(1, -1))[0]
        preds.append(next_pred)
        pred_input = np.append(pred_input[1:], [[next_pred]], axis=0)
    preds_actual = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    st.line_chart(preds_actual)
    results['LightGBM'] = preds_actual
    metrics['LightGBM'] = mean_squared_error(data['close'][-pred_steps:], preds_actual) if len(data) > pred_steps else np.nan

# --- CatBoost ---
with tabs[6]:
    st.markdown("#### CatBoost Forecast")
    model_cat = CatBoostRegressor(verbose=0, iterations=100)
    model_cat.fit(X, y)
    pred_input = data_scaled[-60:]
    preds = []
    for _ in range(pred_steps):
        next_pred = model_cat.predict(pred_input.reshape(1, -1))[0]
        preds.append(next_pred)
        pred_input = np.append(pred_input[1:], [[next_pred]], axis=0)
    preds_actual = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    st.line_chart(preds_actual)
    results['CatBoost'] = preds_actual
    metrics['CatBoost'] = mean_squared_error(data['close'][-pred_steps:], preds_actual) if len(data) > pred_steps else np.nan

# --- Random Forest ---
with tabs[7]:
    st.markdown("#### Random Forest Forecast")
    model_rf = RandomForestRegressor(n_estimators=100)
    model_rf.fit(X, y)
    pred_input = data_scaled[-60:]
    preds = []
    for _ in range(pred_steps):
        next_pred = model_rf.predict(pred_input.reshape(1, -1))[0]
        preds.append(next_pred)
        pred_input = np.append(pred_input[1:], [[next_pred]], axis=0)
    preds_actual = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    st.line_chart(preds_actual)
    results['Random Forest'] = preds_actual
    metrics['Random Forest'] = mean_squared_error(data['close'][-pred_steps:], preds_actual) if len(data) > pred_steps else np.nan

# --- SVR ---
with tabs[8]:
    st.markdown("#### Support Vector Regressor (SVR) Forecast")
    model_svr = SVR()
    model_svr.fit(X, y)
    pred_input = data_scaled[-60:]
    preds = []
    for _ in range(pred_steps):
        next_pred = model_svr.predict(pred_input.reshape(1, -1))[0]
        preds.append(next_pred)
        pred_input = np.append(pred_input[1:], [[next_pred]], axis=0)
    preds_actual = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    st.line_chart(preds_actual)
    results['SVR'] = preds_actual
    metrics['SVR'] = mean_squared_error(data['close'][-pred_steps:], preds_actual) if len(data) > pred_steps else np.nan

# --- Ridge Regression ---
with tabs[9]:
    st.markdown("#### Ridge Regression Forecast")
    model_ridge = Ridge()
    model_ridge.fit(X, y)
    pred_input = data_scaled[-60:]
    preds = []
    for _ in range(pred_steps):
        next_pred = model_ridge.predict(pred_input.reshape(1, -1))[0]
        preds.append(next_pred)
        pred_input = np.append(pred_input[1:], [[next_pred]], axis=0)
    preds_actual = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    st.line_chart(preds_actual)
    results['Ridge'] = preds_actual
    metrics['Ridge'] = mean_squared_error(data['close'][-pred_steps:], preds_actual) if len(data) > pred_steps else np.nan

# --- Model Comparison Table ---
st.subheader(":trophy: Model Performance Comparison (MSE)")
perf_df = pd.DataFrame({
    'Model': list(metrics.keys()),
    'MSE': [metrics[m] for m in metrics]
})
st.dataframe(perf_df.sort_values('MSE'))

st.success("✅ All forecasts generated successfully!")

