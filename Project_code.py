# %% [markdown]
# ## Introduction:
# In the realm of financial markets, understanding the patterns and dynamic of stock prices is of paramount importance in order to blossom in this realm.  
# The stock price of a company reflects not only its intrinsic value but also the collective perception of its future prospects. In this context, the analysis of stock price time series data becomes a crucial endeavor, offering insights into market trends, investor behavior, and potential forecasting opportunities.  
#   
# For our time series analysis project, we have chosen to delve into the world of Apple Inc., one of the most iconic and influential companies in the technology field. the dataset we have selected spance over a long period of time, capturing the rise and the fluctuations in Apple's stock price over time.  
#   
# In this part of the project, we will conduct a comprehensive preliminary analysis of the Apple stock price dataset. We will present the patterns, trends and seasonality ingerent in the data. we will be doing this through graph and visualization.  
#   
# The dataset's shape is (10468, 7), where the columns are Date, Open, High, Low, Close, Adj Close, Volume.

# %%
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations

# Visualization libraries
import matplotlib.pyplot as plt  # For basic plotting
import seaborn as sns  # For heatmap and enhanced visualizations
from matplotlib import dates as mpl_dates  # For handling dates in plots

# Plotly for interactive visualizations
import plotly.express as px
import plotly.graph_objects as go

# Additional Plotly utilities
from plotly.subplots import make_subplots

# For 3D plotting in Matplotlib
from mpl_toolkits.mplot3d import Axes3D


# %%
df = pd.read_csv('AAPL_EXO.csv')  # Load your dataset

# %%
df.shape

# %%
new_df = df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

# %%
new_df.columns

# %%
new_df.shape

# %%
new_df['Date'] = pd.to_datetime(df['Date'])
new_df.set_index('Date', inplace=True)

# %%
# Convert data types for numerical analysis if not already done
# It's essential for accurate calculations
new_df['Open'] = pd.to_numeric(new_df['Open'], errors='coerce')
new_df['High'] = pd.to_numeric(new_df['High'], errors='coerce')
new_df['Low'] = pd.to_numeric(new_df['Low'], errors='coerce')
new_df['Close'] = pd.to_numeric(new_df['Close'], errors='coerce')
new_df['Adj Close'] = pd.to_numeric(new_df['Adj Close'], errors='coerce')
new_df['Volume'] = pd.to_numeric(new_df['Volume'], errors='coerce')

# Calculate descriptive statistics
descriptive_stats = new_df.describe()

# Display the descriptive statistics
print(descriptive_stats)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Decompose the time series
decomposition = seasonal_decompose(new_df['Close'], model='additive', period=365) # or 'multiplicative', depending on your data

# Plot the decomposed components of the time series
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 10)) # You can adjust the size as needed

# Plot the original time series, trend, seasonal, and residual components
decomposition.observed.plot(ax=ax1, title='Original')
ax1.set_ylabel('Observed')

decomposition.trend.plot(ax=ax2, title='Trend')
ax2.set_ylabel('Trend')

decomposition.seasonal.plot(ax=ax3, title='Seasonality')
ax3.set_ylabel('Seasonal')

decomposition.resid.plot(ax=ax4, title='Residuals')
ax4.set_ylabel('Resid')

# Adjust layout
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Original:
# The first plot shows the original time series data for the AAPL closing stock prices. It demonstrates a long-term upward trend with some fluctuations over time. There is a notable acceleration in price increase in the latter part of the series, indicating periods of rapid growth.
# 
# ### Trend: 
# The second plot isolates the trend component from the time series. It smoothes out the short-term fluctuations and highlights the long-term movement. The upward trend is gradual at first and then becomes more pronounced after 2000, reflecting the company's growth and the market's increasing valuation of Apple over time.
# 
# ### Seasonality: 
# The third plot shows the seasonal component of the time series. It exhibits the regular and periodic fluctuations within the time series data that repeat over a specific interval. In this case, the fluctuations seem quite consistent, suggesting a stable seasonal pattern. The scale of seasonality is relatively small compared to the overall trend, which is typical for stock data that doesn't typically have strong seasonality like retail or agricultural sectors might.
# 
# ### Residuals: 
# The final plot illustrates the residuals, which are the time series after the trend and seasonal components have been removed. This component represents the irregular or random noise in the data. Here the residuals seem fairly consistent in variance over time, with some bursts of increased variability. These could be due to external factors or events not accounted for by the seasonal and trend components, or they could indicate periods of market instability.
# 
# In summary, the plots indicate a strong long-term growth in the AAPL stock price with some periods of increased volatility. The stable seasonal component suggests regular patterns, possibly related to business cycles, product release schedules, or other recurring events. The residuals do not show any pronounced patterns, indicating that the trend and seasonal components have captured most of the systematic variation in the time series.

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(new_df['Close'], lags=50)
plt.show()

plot_pacf(new_df['Close'], lags=50)
plt.show()

# %%
plt.plot(new_df.index, new_df['Close'])
plt.plot(new_df.index, new_df['Close'].diff())

plot_acf(new_df['Close'].diff().dropna(), lags=20)
plt.show()

plot_pacf(new_df['Close'].diff().dropna(), lags=50)
plt.show()

# %% [markdown]
# ### Partial Autocorrelation Function (PACF) Plot:
# 
# This plot shows the partial correlation of a time series with its own lagged values, controlling for the values of the time series at all shorter lags. It helps identify the order of an autoregressive (AR) model.
# The PACF plot here has a significant spike at lag 1 and possibly at lag 2, and then it cuts off sharply, which is indicative of an AR(1) or AR(2) process.
# 
# ### Autocorrelation Function (ACF) Plot:
# 
# The ACF plot shows the correlation between the time series and its lagged values.
# The ACF plot provided has a gradual decline in correlation values as the lag increases. This pattern typically suggests that the underlying time series is not stationary and may contain a trend or seasonal component. The slow decay also indicates that an AR model might be appropriate.
# Based on these observations, here are the conclusions:
# 
# Non-Stationarity: The ACF plot indicates non-stationarity due to the gradual decline in correlation values rather than a sharp cut-off. Before fitting a model, you might need to difference the series to make it stationary or consider detrending or deseasonalizing the data.
# 
# Model Suggestion: The PACF plot suggests that an AR(1) or AR(2) model may be appropriate for the data, as indicated by the significant spikes at lags 1 and possibly 2.

# %%
rolling_window = 12 # Example for monthly rolling window
new_df['rolling_mean'] = new_df['Close'].rolling(window=rolling_window).mean()
new_df['rolling_std'] = new_df['Close'].rolling(window=rolling_window).std()

plt.figure(figsize=(14, 7))
plt.plot(new_df.index, new_df['Close'], label='Close Price')
plt.plot(new_df.index, new_df['rolling_mean'], label='Rolling Mean')
plt.plot(new_df.index, new_df['rolling_std'], label='Rolling Std Dev', color='red')
plt.legend()
plt.show()


# %% [markdown]
# #### Trend: 
# The closing price shows a clear upward trend over time. There are periods of relative stability as well as periods of rapid increase, particularly notable in the latter part of the chart.
# 
# #### Volatility: 
# The rolling standard deviation appears to increase with time, especially during the periods when the stock price also increases significantly. This indicates higher volatility during these periods.
# 
# #### Mean: 
# The rolling mean, which is a moving average, increases over time, tracking closely with the upward trend of the closing prices. This suggests a persistent long-term upward trend in the stock price.
# 
# #### Stationarity: 
# The fact that both the rolling mean and rolling standard deviation increase over time suggests that the time series is not stationary. A stationary time series has constant mean and variance over time. Non-stationarity is an important aspect to consider when modeling time series data as most time series models require the data to be stationary.
# 
# #### Potential Transformations: 
# Given the non-stationary nature of the data, differencing or logarithmic transformation might be required to stabilize the mean and variance before fitting a time series model.
# 
# #### Periods of Rapid Change: 
# There are certain periods where the stock price increases rapidly, potentially corresponding to key events or product launches. These would be interesting to investigate further, perhaps with an event study.
# 
# #### Long-Term Investment: 
# For long-term investors, the general upward trend might be interpreted positively, but one should also consider the increasing volatility and potential risks associated with it.
# 
# In conclusion, while the historical stock price data shows a strong upward trend suggesting a good long-term performance, the increasing volatility indicates that there may be higher risks, particularly in the short term. Before using this data for forecasting, it would be crucial to stabilize the variance and possibly detrend the series to ensure any models applied are appropriate and the resulting forecasts are reliable.

# %%
plt.figure(figsize=(14, 7))
plt.plot(new_df.index, df['Close'], label='Apple Stock Price')
plt.title('Apple Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()

# %%
# Ensure the index is a datetime type for resampling
new_df.index = pd.to_datetime(new_df.index)

# Calculate annual returns based on 'Adj Close'
annual_returns = new_df['Adj Close'].resample('Y').last().pct_change()

# %% [markdown]
# ##  High-Low Price Range Each Year
# Showcasing the range of stock prices within each year can illustrate volatility and trading opportunities.

# %%
# Resample data annually to find max and min prices
annual_high = new_df['High'].resample('Y').max()
annual_low = new_df['Low'].resample('Y').min()

plt.figure(figsize=(14, 7))
plt.fill_between(annual_high.index, annual_low, annual_high, color='lightgray', alpha=0.5, label='High-Low Range')
plt.plot(annual_high.index, annual_high, label='Annual High', color='green')
plt.plot(annual_low.index, annual_low, label='Annual Low', color='red')
plt.title('Annual High-Low Price Range of Apple Stock')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# %% [markdown]
# ## Moving Averages
# Moving averages are a foundational analysis tool, smoothing out price data over a specific period and helping to identify trends.

# %%
# Calculate 50-day and 200-day moving averages
new_df['50_MA'] = new_df['Adj Close'].rolling(window=50).mean()
new_df['200_MA'] = new_df['Adj Close'].rolling(window=200).mean()

plt.figure(figsize=(14, 7))
plt.plot(new_df.index, new_df['Adj Close'], label='Adjusted Close Price', color='lightblue')
plt.plot(new_df.index, new_df['50_MA'], label='50-Day Moving Average', color='orange')
plt.plot(new_df.index, new_df['200_MA'], label='200-Day Moving Average', color='magenta')
plt.title('Apple Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.show()

# %% [markdown]
# The chart shows Apple's stock price exhibiting a long-term uptrend, consistently above the 200-day moving average. Fluctuations and crossovers between the 50-day and 200-day moving averages hint at changing short-term trends and potential buy or sell signals, while the distance between the moving averages reflects varying volatility. The overall pattern suggests robust growth with periods of correction towards the mean.

# %% [markdown]
# ## PART 2 - Model Fitting

# %% [markdown]
# After exploring the data and trying to find patterns, we decided to delve into predicting the Close price of the apple stock. Following previous exploration, we suggest that the model which best fits the Close price could be AR(2) or AR(1), but in order to find the best model we're going to test different SARIMA models and also Prophet(Below - every model and it's parameters in addition to the explanation of choice). Also, after we conducted research about the relevancy of the years in our dataset, we conculded that the behaviour of the stock Close price before the year 2007 (The First Release of the iPhone) is not relevant since the Close prices was significantly low compared to today's stock price, So we chose to consider the data from a year before (2006) and on.

# %%
new_df

# %%
# Adjust Relevent Data rows
pred_data = new_df[(new_df.index.year >= 2006) & (new_df.index <= '2022-06-17')]

# %%
# A dataset for comparing forecasts for current date interval
# not_clean_current_df = pd.read_csv('AAPL_2024.csv')
not_clean_current_df = new_df[new_df.index >= '2022-06-17']
current_df = not_clean_current_df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
# current_df['Date'] = pd.to_datetime(current_df['Date'])
# current_df.set_index('Date',inplace=True)

# %%
new_df = new_df[new_df.index <= '2022-06-17']

# %% [markdown]
# ### Components for our first model - ARIMA(1,1,1) :
# Due to the sharp cutoff in PACF, we suggest AR(1) model then we set p=1. The quick drop-off of ACF and preliminary analysis could suggest a MA(1) could be tested, so we set q=1. Then we set differencing to 1 due to non-stationarity.

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Train ARIMA model with parameters (p=1, d=1, q=0)
order = (1, 1, 1)  # ARIMA parameters: (p, d, q)
# seasonal_order = (1, 1, 1, 10)  # Seasonal parameters: (P, D, Q, S)

# data column we're going to predict
data = pred_data['Close']

# model = ARIMA(pred_data['Close'], order=order)
sarima10_model = SARIMAX(data, order=order)

fit_model = sarima10_model.fit()

# Print model summary
print(fit_model.summary())

# Plot residuals
fit_model.plot_diagnostics(figsize=(15, 10))
plt.show()

# Make predictions
forecast = fit_model.forecast(steps=10)  # Change steps to the desired forecast horizon

# Print forecast
print("Forecast:", forecast)

# %%
periods = 252*3

forecast = fit_model.get_forecast(steps=periods)
# Get forecast values
forecast_sarima0_values = forecast.predicted_mean

# Get confidence intervals for the forecast
forecast_ci = forecast.conf_int()

# define last date in the dataset
last_date = new_df.index[-1]

# Create a date range for the forecast values, starting the day after the last_date
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')

# Assume forecast_values is a Series or array from your SARIMA model forecast
forecast_sarima0_values.index = forecast_index

# Getting forecast confidence intervals
forecast_ci.index = forecast_index

# Plotting
plt.figure(figsize=(15, 10))
plt.plot(pd.concat([new_df['Close'],current_df['Close']],axis=0), label='Historical Daily Close Price')
plt.plot(forecast_sarima0_values.index, forecast_sarima0_values, color='red', label='Forecast')
plt.fill_between(forecast_sarima0_values.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.3)

plt.title('Forecast vs Historical')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# %% [markdown]
# #### Results Analysis for first model:
# Not a great performance for the model according to the analysis table, we can spot a relatively high log likelihood of the model and and high AIC and BIC, which indicates a poor performance

# %% [markdown]
# ### Components for our second model - SARIMA(1,1,1)(1,1,1)[63] :
# #### Non-seasonal
# Since the PACF showed a sharp cutoff after lag 1, that suggest AR(1) model then we set p=1. The quick drop-off of ACF could suggest a MA(1) could be tested, particularly if after differencing the ACF shows a spike at lag 1, the q=1. Then we set differencing to 1 due to non-stationarity.
# #### Seasonal:
# In stocks market and for daily information, we want to set seasonality to be the number of quarterly trading days in a year which is 252//4 = 63 as mentioned in this [page](https://en.wikipedia.org/wiki/Trading_day) and the rest of the data would be the same as the non-Seasonal.

# %%
# With SEASONALITY
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Setting SARIMAX according to the conclusions mentioned earlier
data = pred_data['Close']
sarima1_model = SARIMAX(data,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 252//4),
                enforce_stationarity=False,
                enforce_invertibility=False)
# enforce_stationarity=False and enforce_invertibility=False are
# helpful during model exploration, especially when the model diagnostics suggest non-invertibility or non-stationarity issues with the parameters
sarima1_results = sarima1_model.fit(disp=True)

# %%
print(sarima1_results.summary())

# %%
periods = 252*3

forecast = sarima1_results.get_forecast(steps=periods)
# Get forecast values
forecast_sarima1_values = forecast.predicted_mean

# Get confidence intervals for the forecast
forecast_ci = forecast.conf_int()

# define last date in the dataset
last_date = new_df.index[-1]

# Create a date range for the forecast values, starting the day after the last_date
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')

# Assume forecast_values is a Series or array from your SARIMA model forecast
forecast_sarima1_values.index = forecast_index

# Getting forecast confidence intervals
forecast_ci.index = forecast_index

# Plotting
plt.figure(figsize=(15, 10))
# plt.plot(new_df['Close'], label='Historical Daily Close Price')
plt.plot(pd.concat([new_df['Close'],current_df['Close']],axis=0), label='Historical Daily Close Price')
plt.plot(forecast_sarima1_values.index, forecast_sarima1_values, color='red', label='Forecast')
plt.fill_between(forecast_sarima1_values.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.3)

plt.title('Forecast vs Historical')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# %%
print("Numerical Values Of our Second model forecast:")
print(forecast_sarima1_values)

# %% [markdown]
# #### Results Analysis of Second model:
# We can sense a slight improvement with this model in relation to the first one, here we can see the prediction being more logical. On the other hand, both AIC and BIC parameters are better (lower values) than the first model, in addition to the log-likelihood being less than the first model.

# %% [markdown]
# ### Components for our third model - ARIMA(2,0,2) :
# In order to capture more information using the models, we decided to try the ARIMA model with the p=q=2 because of the AR(2) and MA(2) suggestion and d=1 due to non-stationary data. We removed seasonality to check if it'd get better.

# %%
# Without SEASONALITY
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Setting SARIMAX according to the conclusions mentioned earlier
data = pred_data['Close']
sarima2_model = SARIMAX(data,
                order=(2, 0, 2),
                enforce_stationarity=False,
                enforce_invertibility=False)
# enforce_stationarity=False and enforce_invertibility=False are
# helpful during model exploration, especially when the model diagnostics suggest non-invertibility or non-stationarity issues with the parameters
sarima2_results = sarima2_model.fit()

# %%
print(sarima2_results.summary())

# %%
periods = 252*3

forecast = sarima2_results.get_forecast(steps=periods)
# Get forecast values
forecast_sarima2_values = forecast.predicted_mean

# Get confidence intervals for the forecast
forecast_ci = forecast.conf_int()

# define last date in the dataset
last_date = new_df.index[-1]

# Create a date range for the forecast values, starting the day after the last_date
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')

# Assume forecast_values is a Series or array from your SARIMA model forecast
forecast_sarima2_values.index = forecast_index

# Getting forecast confidence intervals
forecast_ci.index = forecast_index

# Plotting
plt.figure(figsize=(15, 10))
plt.plot(pd.concat([new_df['Close'],current_df['Close']],axis=0), label='Historical Daily Close Price')
# plt.plot(new_df['Close'], label='Historical Daily Close Price')
plt.plot(forecast_sarima2_values.index, forecast_sarima2_values, color='red', label='Forecast')
plt.fill_between(forecast_sarima2_values.index, 
                 forecast_ci.iloc[:, 0], 
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.3)

plt.title('Forecast vs Historical')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# %%
print("Numerical Values Of our third model forecast:")
print(forecast_sarima2_values)

# %% [markdown]
# #### Results analysis of the fourth model:
# In terms of numbers, the model is not very much better that other ones, but if we look at the predicted observations, we can see that this model has a very strong understanding of Close price of the stock, which can be benficial when incorporating exogenous variables to this dataset.

# %% [markdown]
# ### Our fourth model: Prophet

# %%
from prophet import Prophet

# Assuming 'data' is the DataFrame with columns 'ds' (datetime) and 'y' (target variable)
# Rename the columns to match Prophet's requirements

data = pred_data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

# Initialize Prophet model
prophet_model = Prophet()

# Fit the model to the data
prophet_model.fit(data)

# Make future predictions
future = prophet_model.make_future_dataframe(periods=100)  # Adjust the number of future periods as needed
forecast = prophet_model.predict(future)

# Plot the forecast
fig = prophet_model.plot(forecast)
fig.legend()

# %%
se = np.square(forecast.loc[:, 'yhat'] - data['y'])
se_mae = np.abs(forecast.loc[:, 'yhat'] - data['y'])

mse = np.mean(se)
mae = np.mean(se_mae)

rmse = np.sqrt(mse)

print('MAE of Prohpet: {}'.format(mae))
print('RMSE of Prophet: {}'.format(rmse))


# Get the in-sample predictions
in_sample_predictions = prophet_model.predict(data)

# Calculate residuals
residuals = data['y'] - in_sample_predictions['yhat']

# Calculate the effective number of parameters (p_eff)
p_eff = prophet_model.params['k'].shape[0]

# Calculate the log likelihood
log_likelihood = -0.5 * (np.log(2 * np.pi * residuals.std()) + (residuals ** 2 / residuals.var()).sum())

# Calculate the penalized log likelihood
penalized_log_likelihood = log_likelihood - p_eff

# Calculate WAIC
waic = -2 * penalized_log_likelihood + 2 * p_eff

print("WAIC:", waic)



# from prophet.diagnostics import cross_validation

# # Perform cross-validation
# cv_results = cross_validation(prophet_model, horizon='100 days')  # Adjust the horizon as needed

# # Calculate WAIC
# waic = -2 * (cv_results['y'] - cv_results['yhat']).mean()
# print("WAIC:", waic)

# %% [markdown]
# #### Prophet Results Analysis:
# Looking at the plot that compares actual vs predicted close stock price, we can sense the greatness of the prophet model. The prediction looks very good, and statistically, RMSE and MAE between the actual and predicted observations is significanlty low, it's an indication of a great performance by the model.

# %% [markdown]
# ## Visual comparison of the predictions of models

# %%
# plotting
plt.figure(figsize=(15,10))
# plt.plot(pd.concat([pred_data['Close'],current_df['Close']],axis=0), label='True Price')
plt.plot(current_df['Close'], label='True Close Price')
# plt.plot(new_forecast.loc[:, 'yhat'], label='Prophet Prediction')
plt.plot(forecast_sarima0_values.index, forecast_sarima0_values, color='red', label='First Model Forecast (ARIMA(1, 1, 1))')
plt.plot(forecast_sarima1_values.index, forecast_sarima1_values, color='orange', label='Second Model Forecast (SARIMA(1, 1, 1)(1, 1, 1)[63])')
plt.plot(forecast_sarima2_values.index, forecast_sarima2_values, color='black', label='Third Model Forecast (ARIMA(2, 0, 2))')
plt.title('True Close Price Vs. Predicted By Our Models')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# %% [markdown]
# # PART 3 - Incorporating an Exogenous Variable

# %% [markdown]
# Stock prices could be significanlty influenced by many indicators, including Apple products sales, new devices launches, GDP growth, interest rates and so on. After conducting research in exploring datasets that could be helpful for our data, we found that [NASDAQ-100 index](https://www.investopedia.com/terms/n/nasdaq100.asp) might be an interesting insight into predicting the Apple stock Close price. (The dataset we found regerading this index is partial in relation of the stock data we're currently working on, so we adjust the dataframes we're working on to fit the index's dataset)

# %%
# not_clean_index_df = pd.read_csv('Jan20_NDXT.csv')  # Load NASDAQ-100 index dataset
# index_df = not_clean_index_df.dropna()

# %%
# index_df.shape

# %%
# index_df.columns

# %%
# index_df['Date'] = pd.to_datetime(index_df['Date'])
# index_df.set_index('Date', inplace=True)

# %%
# Merge the two dataframes
# merged_df = pd.merge(pred_data, index_df, left_index=True, right_index=True)
merged_df = pred_data.dropna()

# %%
# Without SEASONALITY
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Setting SARIMAX with the exogenous variable
data = merged_df['Close']
exog = merged_df['NDXT Close']
sarima2exog_model = SARIMAX(data, exog=exog,
                order=(2, 1, 2),
                enforce_stationarity=False,
                enforce_invertibility=False)
# enforce_stationarity=False and enforce_invertibility=False are
# helpful during model exploration, especially when the model diagnostics suggest non-invertibility or non-stationarity issues with the parameters
sarima2exog_results = sarima2exog_model.fit()

# %%
print(sarima2exog_results.summary())

# %%
# Plot residuals
sarima2exog_results.plot_diagnostics(figsize=(15, 10))
plt.show()

# %% [markdown]
# Revewing the diagnostics of the model with exogenous variable, we can see a huge difference compared to models we explored in the previous section, the log-likelihood (-42.788) is far more better, and AIC= 97 and BIC=133 is much better with a wide margin 

# %% [markdown]
# 

# %%
# With SEASONALITY
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Setting SARIMAX according to the conclusions mentioned earlier
sarima1_model = SARIMAX(data, exog=exog,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 252//4),
                enforce_stationarity=False,
                enforce_invertibility=False)
# enforce_stationarity=False and enforce_invertibility=False are
# helpful during model exploration, especially when the model diagnostics suggest non-invertibility or non-stationarity issues with the parameters
sarima1_results = sarima1_model.fit(disp=True)

# %%
print(sarima1_results.summary())

# %%
# Plot residuals
sarima1_results.plot_diagnostics(figsize=(15, 10))
plt.show()

# %%


# %%
# Assuming 'data' is the DataFrame with columns 'ds' (datetime), 'y' (target variable), and 'exog' (exogenous variable)
# Rename the columns to match Prophet's requirements

data = merged_df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y', 'NDXT Close': 'exog'})

# Initialize Prophet model with exogenous regressor
prophet_model = Prophet()
prophet_model.add_regressor('exog')  # Add the exogenous variable to the model

# Fit the model to the data
prophet_model.fit(data)

forecast = prophet_model.predict(data)

# Plot the forecast
fig = prophet_model.plot(forecast)
fig.legend()

# %%
# Get the in-sample predictions
in_sample_predictions = prophet_model.predict(data)

# Calculate residuals
residuals = data['y'] - in_sample_predictions['yhat']

# Calculate the effective number of parameters (p_eff)
p_eff = prophet_model.params['k'].shape[0]

# Calculate the log likelihood
log_likelihood = -0.5 * (np.log(2 * np.pi * residuals.std()) + (residuals ** 2 / residuals.var()).sum())

# Calculate the penalized log likelihood
penalized_log_likelihood = log_likelihood - p_eff

# Calculate WAIC
waic = -2 * penalized_log_likelihood + 2 * p_eff

print("WAIC:", waic)


