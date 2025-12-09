# streamlit_forecast_app.py
# Streamlit app built to mirror your notebook workflow for commodity forecasting.
# Updated: Automatically loads a bundled Data.xlsx from the app directory (if present)
# so you do not need to upload a file. If the bundled file is missing, the app will
# ask you to upload an Excel file.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import io
import os
import warnings
warnings.filterwarnings('ignore')

# Time series models
from statsmodels.tsa.api import SARIMAX, ETSModel
from pmdarima import auto_arima
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

st.set_page_config(page_title='Commodity Forecasting', layout='wide')
st.title('Commodity Forecasting Dashboard')
st.markdown("""
This app will try to load `Data.xlsx` from the same folder as the app automatically.
If `Data.xlsx` is not found, you can upload your Excel file using the sidebar.

Expected Excel columns: `Date`, `Commodity`, `Close_INR`.
""")

# Sidebar
with st.sidebar:
    st.image("enrich_logo.png")

st.sidebar.header('Data & Model Inputs')
use_uploaded = st.sidebar.checkbox('Force file upload instead of bundled Data.xlsx', value=False)
uploaded_file = st.sidebar.file_uploader('Upload Excel file (.xlsx) with historical data', type=['xlsx','xls'])

freq = st.sidebar.selectbox('Resample frequency', options=['D','W','M'], index=2, format_func=lambda x: {'D':'Daily','W':'Weekly','M':'Monthly'}[x])
model_choice = st.sidebar.selectbox('Model', options=['AutoARIMA','SARIMAX','ETS','Prophet (if installed)'])
periods = st.sidebar.number_input('Forecast periods (horizon)', min_value=1, max_value=120, value=12)
train_ratio = st.sidebar.slider('Train ratio (for backtest)', 0.5, 0.95, 0.8)

# Attempt to load bundled data
@st.cache_data
def load_excel_from_bytes(bytes_io):
    return pd.read_excel(bytes_io, parse_dates=['Date'])

def try_load_bundled(path='Data.xlsx'):
    if os.path.exists(path):
        try:
            df = pd.read_excel(path, parse_dates=['Date'])
            return df
        except Exception as e:
            st.warning(f'Bundled file found but failed to read: {e}')
            return None
    return None

# Decide source
df = None
if not use_uploaded:
    df = try_load_bundled('./Data.xlsx')
    if df is not None:
        st.success('Loaded bundled Data.xlsx from app folder.')

if df is None:
    if uploaded_file is not None:
        try:
            df = load_excel_from_bytes(uploaded_file)
            st.success('Uploaded file loaded successfully')
        except Exception as e:
            st.error(f'Failed to read uploaded file: {e}')
            st.stop()
    else:
        st.warning('No bundled Data.xlsx found in app folder and no file uploaded.Place Data.xlsx next to this app or upload one in the sidebar.')
        st.stop()

# Basic validation
if 'Commodity' not in df.columns or 'Close_INR' not in df.columns or 'Date' not in df.columns:
    st.error('Excel must contain columns: Date, Commodity, Close_INR')
    st.write('Columns found:', list(df.columns))
    st.stop()

# Clean dataframe
df = df.copy()
df['Date'] = pd.to_datetime(df['Date'])

commodity_list = df['Commodity'].unique().tolist()
commodity = st.selectbox('Choose commodity', options=commodity_list)

series = df.loc[df['Commodity']==commodity, ['Date','Close_INR']].dropna().sort_values('Date')
series = series.set_index('Date')

# Resample
if freq == 'D':
    ts = series['Close_INR'].asfreq('D').interpolate()
elif freq == 'W':
    ts = series['Close_INR'].resample('W').mean().interpolate()
else:
    ts = series['Close_INR'].resample('M').mean().interpolate()

st.subheader(f'{commodity} — time series (resampled: {freq})')
st.line_chart(ts.tail(200))

# Train/test split
n = len(ts)
train_n = int(n * train_ratio)
train = ts.iloc[:train_n]
test = ts.iloc[train_n:]

st.write(f'Total points: {n} — Train: {len(train)} — Test: {len(test)}')

# Model functions
@st.cache_data
def fit_auto_arima(train_series, seasonal=False, m=12):
    model = auto_arima(train_series, seasonal=seasonal, m=m, stepwise=True, suppress_warnings=True, error_action='ignore')
    return model

@st.cache_data
def fit_sarimax(train_series, order=(1,1,1), seasonal_order=(0,0,0,0)):
    ms = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = ms.fit(disp=False)
    return res

@st.cache_data
def fit_ets(train_series):
    m = ETSModel(train_series, error='add', trend='add', seasonal=None)
    res = m.fit()
    return res

# Fit & forecast depending on model choice
with st.spinner('Training and forecasting...'):
    forecast_df = None
    metrics = {}
    if model_choice == 'AutoARIMA':
        seasonal_flag = (freq == 'M' or freq == 'W')
        m = fit_auto_arima(train, seasonal=seasonal_flag, m=12 if freq=='M' else (52 if freq=='W' else 7))
        steps = len(test) + periods
        preds = m.predict(n_periods=steps)
        pred_index = pd.date_range(start=train.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=steps, freq=freq)
        preds_series = pd.Series(preds, index=pred_index)
        forecast_df = preds_series.iloc[-periods:].rename('yhat').to_frame()
        if len(test) > 0:
            test_pred = preds_series.iloc[:len(test)]
            mse = mean_squared_error(test, test_pred[:len(test)])
            rmse = np.sqrt(mse)
            metrics = {'MSE':mse, 'RMSE':rmse}
    elif model_choice == 'SARIMAX':
        order=(1,1,1)
        if freq=='M':
            seasonal_order=(1,1,1,12)
        elif freq=='W':
            seasonal_order=(1,1,1,52)
        else:
            seasonal_order=(0,0,0,0)
        res = fit_sarimax(train, order=order, seasonal_order=seasonal_order)
        steps = len(test)+periods
        preds = res.get_prediction(start=len(train), end=len(train)+steps-1)
        pred_mean = preds.predicted_mean
        pred_index = pd.date_range(start=train.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=steps, freq=freq)
        pred_series = pd.Series(pred_mean.values, index=pred_index)
        forecast_df = pred_series.iloc[-periods:].rename('yhat').to_frame()
        if len(test) > 0:
            test_pred = pred_series.iloc[:len(test)]
            mse = mean_squared_error(test, test_pred[:len(test)])
            rmse = np.sqrt(mse)
            metrics = {'MSE':mse, 'RMSE':rmse}
    elif model_choice == 'ETS':
        res = fit_ets(train)
        steps = len(test)+periods
        preds = res.forecast(steps)
        pred_index = pd.date_range(start=train.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=steps, freq=freq)
        pred_series = pd.Series(preds, index=pred_index)
        forecast_df = pred_series.iloc[-periods:].rename('yhat').to_frame()
        if len(test) > 0:
            test_pred = pred_series.iloc[:len(test)]
            mse = mean_squared_error(test, test_pred[:len(test)])
            rmse = np.sqrt(mse)
            metrics = {'MSE':mse, 'RMSE':rmse}
    else: # Prophet
        if not PROPHET_AVAILABLE:
            st.error('Prophet is not installed in this environment. Choose another model or install prophet.')
            st.stop()
        df_train = train.reset_index().rename(columns={'Date':'ds','Close_INR':'y'})
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=len(test)+periods, freq=freq)
        fc = m.predict(future)
        pred = fc.set_index('ds')['yhat']
        pred_index = pred.index[pred.index>train.index[-1]]
        pred_series = pred.loc[pred_index]
        forecast_df = pred_series.iloc[-periods:].rename('yhat').to_frame()
        if len(test) > 0:
            test_pred = pred_series.iloc[:len(test)]
            mse = mean_squared_error(test, test_pred[:len(test)])
            rmse = np.sqrt(mse)
            metrics = {'MSE':mse, 'RMSE':rmse}

# Display results
st.subheader('Forecast & Evaluation')
if forecast_df is None:
    st.error('Forecast generation failed')
    st.stop()

# Plot history + forecast
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(ts.index, ts.values, label='history')
fc_index = forecast_df.index
ax.plot(fc_index, forecast_df['yhat'].values, label='forecast', linestyle='--')
ax.axvline(x=train.index[-1], color='gray', linestyle=':')
ax.set_title(f'{commodity} forecast ({model_choice})')
ax.set_ylabel('Close_INR')
ax.legend()
st.pyplot(fig)

# Show metrics
if metrics:
    st.metric('RMSE', f"{metrics.get('RMSE'):.4f}")
    st.write('MSE:', metrics.get('MSE'))

st.subheader('Forecast table (next periods)')
st.dataframe(forecast_df.reset_index().rename(columns={'index':'Date'}))

# Download CSV
csv_bytes = forecast_df.reset_index().rename(columns={'index':'Date'}).to_csv(index=False).encode('utf-8')
st.download_button('Download forecast CSV', data=csv_bytes, file_name=f'{commodity}_forecast.csv', mime='text/csv')

st.markdown('---')
st.caption('''Notes: 
- AutoARIMA requires pmdarima. 
- Prophet is optional and must be installed in the environment. 
- This app uses simple defaults (orders) for SARIMAX/ETS when the user chooses them; for production you should grid-search hyperparameters or reuse the exact orders you used in your notebook.''')
