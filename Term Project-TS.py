import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')

# ==============================================
# Data Preprocessing and Summary Statistics
# ==============================================

file_path = '/Users/srivarshini/Downloads/archive/climate_data_ts.csv'
climate_data = pd.read_csv(file_path)

date_format = '%Y-%m-%d'
climate_data['date'] = pd.to_datetime(climate_data['date'], format=date_format)

climate_data.set_index('date', inplace=True)

# Display basic information about the dataset
print("===== Data Set Information =====")
print(climate_data.info())

# Display the first few rows of the dataset
print("\n===== First Few Rows of the Dataset =====")
print(climate_data.head())
print(climate_data.describe())

# Renaming columns using a dictionary
new_column_names = {
    'Tn': 'Min_Temp',
    'Tx': 'Max_Temp',
    'Tavg': 'Avg_Temp',
    'RH_avg': 'Avg_Relative_Humidity',
    'RR': 'Rainfall',
    'ss': 'Sunshine_Duration',
    'ff_x': 'Max_Wind_Speed',
    'ddd_x': 'Max_Wind_Direction',
    'ff_avg': 'Avg_Wind_Speed',
    'ddd_car': 'Wind_Direction',
    'station_id': 'Station_ID'
}

climate_data.rename(columns=new_column_names, inplace=True)

# Check for missing values
print("\n===== Missing Values =====")
print(climate_data.isnull().sum())

# Convert 'Wind_Direction_Cardinal' to a categorical type
climate_data['Wind_Direction'] = climate_data['Wind_Direction'].astype('category')
climate_data['Wind_Direction'] = climate_data['Wind_Direction'].fillna(climate_data['Wind_Direction'].mode()[0])

# Calculate mean only for numeric columns
numeric_cols = climate_data.select_dtypes(include=[np.number]).columns.tolist()
numeric_means = climate_data[numeric_cols].mean()

# Fill missing values for numeric columns with their respective means
climate_data[numeric_cols] = climate_data[numeric_cols].fillna(numeric_means)

# Fill categorical columns with the mode (most frequent value)
categorical_cols = climate_data.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    climate_data[col] = climate_data[col].fillna(climate_data[col].mode()[0])


print("\n===== Missing Values after Data Cleaning =====")
print(climate_data.isnull().sum())

# Optimizing Data Types
climate_data['station_id'] = climate_data['Station_ID'].astype('int32')
for column in ['Min_Temp', 'Max_Temp', 'Avg_Temp', 'Avg_Relative_Humidity', 'Rainfall', 'Sunshine_Duration', 'Max_Wind_Speed', 'Max_Wind_Direction', 'Avg_Wind_Speed']:
    climate_data[column] = climate_data[column].astype('float32')
print(climate_data.head())

# Check the frequency of the data
print("All columns present in climate_data", climate_data.columns)

# Check the time range covered by the data
print("\n===== Time Range =====")
print(f"Start Date: {climate_data.index.min()}")
print(f"End Date: {climate_data.index.max()}")

# Check for duplicates
duplicate_rows = climate_data[climate_data.duplicated()]
print("\n===== Duplicate Rows =====")
print(duplicate_rows)

# Display the first few rows of the preprocessed dataset
print("\n===== First Few Rows of the Preprocessed Dataset =====")
print(climate_data.head())


# ==============================================
# Exploratory Data Analysis
# ==============================================

# Histogram for all numerical features
climate_data[numeric_cols].hist(bins=15, figsize=(15, 10))
plt.show()

# Correlation matrix
corr_matrix = climate_data[numeric_cols].corr()
print(corr_matrix)

# Heatmap for correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Plot for Wind Direction
if 'Wind_Direction' in climate_data.columns:
    sns.countplot(x='Wind_Direction', data=climate_data)
    plt.title('Frequency Distribution of Wind Directions')
    plt.show()


# Subplots for Min_Temp, Max_Temp and Avg_Temp
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(climate_data.index, climate_data['Min_Temp'], label='Minimum Daily Temperature', color='orange')
plt.title('Minimum Daily Temperature')
plt.xlabel('date')
plt.ylabel('Temperature (°C)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(climate_data.index, climate_data['Max_Temp'], label='Maximum Daily Temperature', color='purple')
plt.title('Maximum Daily Temperature')
plt.xlabel('date')
plt.ylabel('Temperature (°C)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(climate_data.index, climate_data['Avg_Temp'], label='Average Temperature', color='brown')
plt.title('Average Temperature')
plt.xlabel('date')
plt.ylabel('Temperature (°C)')
plt.legend()

plt.tight_layout()
plt.show()

# Humidity Analysis - Scatter plot for Humidity vs Temperature
sns.scatterplot(x='Avg_Temp', y='Avg_Relative_Humidity', data=climate_data)
plt.title('Average Temperature vs Relative Humidity')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Average Relative Humidity (%)')
plt.show()

# Rainfall Patterns
plt.plot(climate_data.index, climate_data['Rainfall'], label='Rainfall in mm', color='teal')
plt.title('Rainfall Patterns')
plt.xlabel('date')
plt.ylabel('Rainfall')
plt.legend()
plt.show()

# Wind Speed vs. Temperature
sns.scatterplot(x='Avg_Wind_Speed', y='Avg_Temp', data=climate_data)
plt.title('Wind Speed vs. Average Temperature')
plt.xlabel('Average Wind Speed (m/s)')
plt.ylabel('Average Temperature (°C)')
plt.show()

# Minimum Temperature vs. Sunshine
sns.scatterplot(x='Sunshine_Duration', y='Min_Temp', data=climate_data)
plt.title('Minimum Temperature vs. Sunshine')
plt.xlabel('Sunshine Duration')
plt.ylabel('Minimum Temperature recorded')
plt.show()

# Wind Patterns
climate_data[['Max_Wind_Speed', 'Avg_Wind_Speed']].plot(figsize=(15, 5))
plt.title('Wind Speed Trends over Time')
plt.ylabel('Wind Speed (km/h)')
plt.xlabel('Date')
plt.show()

# ============================
# Dealing with outliers
# ============================

# Plot the boxplot for the target variable
plt.figure(figsize=(10, 6))
sns.boxplot(x=climate_data['Avg_Temp'])
plt.title('Boxplot of Average Temperature')
plt.xlabel('Average Temperature')
plt.show()

# Calculate the IQR
Q1 = climate_data['Avg_Temp'].quantile(0.25)
Q3 = climate_data['Avg_Temp'].quantile(0.75)
IQR = Q3 - Q1

# Define thresholds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Replace outliers with NaN
climate_data['Avg_Temp_Outliers_Removed'] = climate_data['Avg_Temp'].apply(
    lambda x: np.nan if x < lower_bound or x > upper_bound else x
)

# Fill NaNs with the mean
climate_data['Avg_Temp_Filled'] = climate_data['Avg_Temp_Outliers_Removed'].fillna(climate_data['Avg_Temp'].mean())

plt.figure(figsize=(12, 6))
plt.plot(climate_data['Avg_Temp'], marker='o', linestyle='-', label='Original')
plt.plot(climate_data['Avg_Temp_Outliers_Removed'], marker='o', linestyle='--', label='Outliers Removed')
plt.title('Temperature with Outliers Removed')
plt.legend()

plt.tight_layout()
plt.show()

# ==============================================
# Stationarity Tests
# ==============================================

# function to calculate rolling mean and variance
def cal_rolling_mean_var(df, column_name, window_size):
    rolling_mean = df[column_name].rolling(window=window_size).mean()
    rolling_variance = df[column_name].rolling(window=window_size).var()

    return rolling_mean, rolling_variance

window_size = 4

# rolling mean and rolling variance for 'Avg_Temp'
temp_rolling_mean, temp_rolling_variance = cal_rolling_mean_var(climate_data, 'Avg_Temp', window_size)

# Plot the rolling mean and rolling variance
plt.figure(figsize=(14, 8))

# Rolling Mean Subplot
plt.subplot(2, 1, 1)
plt.plot(climate_data.index, temp_rolling_mean, label='Avg Temp Rolling Mean', color='blue')
plt.title('Rolling Mean of Avg Temp')
plt.xlabel('date')
plt.ylabel('Rolling Mean Temperature')
plt.legend()

# Rolling Variance Subplot
plt.subplot(2, 1, 2)
plt.plot(climate_data.index, temp_rolling_variance, label='Avg Temp Rolling Variance', color='green')
plt.title('Rolling Variance of Avg Temp')
plt.xlabel('date')
plt.ylabel('Rolling Variance Temperature')
plt.legend()

plt.tight_layout()
plt.show()

# Plot Time Series data for Avg Temp
plt.figure(figsize=(14, 7))
plt.plot(climate_data['Avg_Temp'], label='Average Temperature')
plt.title('Average Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.show()

# ============================
# Test for Stationarity
# ============================

# ADF Test
from statsmodels.tsa.stattools import adfuller, kpss

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    return result
# ADF Test for Average Temperature
print("ADF Test for Average Temperature:\n")
result_ADF_cal = ADF_Cal(climate_data['Avg_Temp'])

if result_ADF_cal[1] <= 0.05:
    print("Reject the null hypothesis. The data is stationary.")
else:
    print("Fail to reject the null hypothesis. The data is non-stationary.")

def cal_kpss_test(timeseries):
    timeseries = timeseries.fillna(timeseries.mode())
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)
    return kpss_output

# KPSS test
print("KPSS Test for Target Variable - Average Temperature:")
result_kpss = cal_kpss_test(climate_data['Avg_Temp'])

if result_kpss['p-value'] <= 0.05:
    print("Reject the null hypothesis. The data is non-stationary.")
else:
    print("Fail to reject the null hypothesis. The data is stationary.")

# differencing
climate_data['Avg_Temp_Diff'] = climate_data['Avg_Temp'] - climate_data['Avg_Temp'].shift(1)
climate_data['Avg_Temp_Diff'].dropna()

# ===============================
# ADF Test for Differenced Average Temperature
# ===============================

print("ADF Test for Differenced Average Temperature:\n")
result_ADF_cal_diff = ADF_Cal(climate_data['Avg_Temp_Diff'].dropna())

if result_ADF_cal_diff[1] <= 0.05:
    print("Reject the null hypothesis. The data is stationary.")
else:
    print("Fail to reject the null hypothesis. The data is non-stationary.")

# KPSS Test for Differenced Average Temperature
print("KPSS Test for Differenced Average Temperature:")
result_kpss_diff = cal_kpss_test(climate_data['Avg_Temp_Diff'].dropna())

if result_kpss_diff['p-value'] <= 0.05:
    print("Reject the null hypothesis. The data is non-stationary.")
else:
    print("Fail to reject the null hypothesis. The data is stationary.")


temp_rolling_mean_diff, temp_rolling_variance_diff = cal_rolling_mean_var(climate_data, 'Avg_Temp_Diff', window_size)

# Plot for transformed data
plt.figure(figsize=(14, 8))

# Rolling Mean Subplot
plt.subplot(2, 1, 1)
plt.plot(climate_data.index, temp_rolling_mean_diff, label='Avg Temp Rolling Mean', color='blue')
plt.title('Rolling Mean of Avg Temp after differencing')
plt.xlabel('date')
plt.ylabel('Rolling Mean Temperature')
plt.legend()

# Rolling Variance Subplot
plt.subplot(2, 1, 2)
plt.plot(climate_data.index, temp_rolling_variance_diff, label='Avg Temp Rolling Variance', color='green')
plt.title('Rolling Variance of Avg Temp after differencing')
plt.xlabel('date')
plt.ylabel('Rolling Variance Temperature')
plt.legend()

plt.tight_layout()
plt.show()


# Plot the ACF and PACF for Avg Temp
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot the ACF
plot_acf(climate_data['Avg_Temp_Diff'], lags=40, ax=ax[0])
ax[0].set_title('ACF Transformed Data')

# Plot the PACF
plot_pacf(climate_data['Avg_Temp_Diff'], lags=40, ax=ax[1])
ax[1].set_title('PACF Transformed Data')

plt.tight_layout()
plt.show()

# ============================
# Time  Series Decomposition
# ============================

from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

seasonal_period = 365

# Apply STL decomposition
stl = STL(climate_data['Avg_Temp'], period=seasonal_period, robust=True)
result = stl.fit()
print(result)
# Extract components
trend_component = result.trend
seasonal_component = result.seasonal
residual_component = result.resid

# Detrended Data
detrended_data = climate_data['Avg_Temp'] - trend_component

# Seasonally Adjusted Data
seasonally_adjusted_data = climate_data['Avg_Temp'] - seasonal_component

# Plotting
plt.figure(figsize=(12, 8))

# Original Data
plt.subplot(3, 1, 1)
plt.plot(climate_data['Avg_Temp'], label='Original Data', color='blue')
plt.title('Original Data')
plt.legend()

# Detrended Data
plt.subplot(3, 1, 2)
plt.plot(detrended_data, label='Detrended Data', color='orange')
plt.title('Detrended Data')
plt.legend()

# Seasonally Adjusted Data
plt.subplot(3, 1, 3)
plt.plot(seasonally_adjusted_data, label='Seasonally Adjusted Data', color='green')
plt.title('Seasonally Adjusted Data')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate strength of seasonality
seasonal_strength = result.seasonal.std() / climate_data['Avg_Temp'].std()
print(f'Strength of seasonality: {seasonal_strength:.3f}')

# Calculate strength of trend
trend_strength = result.trend.std() / climate_data['Avg_Temp'].std()
print(f'Strength of trend: {trend_strength:.3f}')

# ============================
# Ljung-Box and Normality Test
# ============================

from scipy.stats import normaltest
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Perform the Ljung-Box test on residuals
ljung_box_result = acorr_ljungbox(result.resid, lags=[10], return_df=True)
print('Ljung-Box test results:')
print(ljung_box_result)

# Normality Test
stat, p = normaltest(result.resid)
print(f'Statistics={stat:.3f}, p={p:.3f}')
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

# Plot the ACF and PACF for the residuals
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot the ACF
plot_acf(result.resid, lags=40, ax=ax[0])
ax[0].set_title('ACF of Residuals')

# Plot the PACF
plot_pacf(result.resid, lags=40, ax=ax[1])
ax[1].set_title('PACF of Residuals')

plt.tight_layout()
plt.show()

# ============================
# Train test split
# ============================
train_data, test_data = train_test_split(climate_data['Avg_Temp_Filled'], test_size=0.2, shuffle=False)


# ============================
# Holt-Winters Method
# ============================

hw_additive_model = ExponentialSmoothing(train_data,
                                         trend='add',
                                         seasonal='add',
                                         seasonal_periods=365).fit()

# Forecasting using the additive model
hw_additive_forecast = hw_additive_model.forecast(steps=len(test_data))

# Plotting the forecasts
plt.figure(figsize=(15, 6))
plt.plot(train_data.index, train_data, label='Train Data', color='blue')
plt.plot(test_data.index, test_data, label='Test Data', color='green')
plt.plot(hw_additive_forecast.index, hw_additive_forecast, label='Holt-Winters Additive Forecast', color='red')
plt.title('Holt-Winters Additive Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.show()

# Applying Holt-Winters method with multiplicative seasonality
hw_multiplicative_model = ExponentialSmoothing(train_data,
                                               trend='add',
                                               seasonal='mul',
                                               seasonal_periods=365).fit()

# Forecasting using the multiplicative model
hw_multiplicative_forecast = hw_multiplicative_model.forecast(steps=len(test_data))

# Plotting the forecasts
plt.figure(figsize=(15, 6))
plt.plot(train_data.index, train_data, label='Train Data', color='blue')
plt.plot(test_data.index, test_data, label='Test Data', color='green')
plt.plot(hw_multiplicative_forecast.index, hw_multiplicative_forecast, label='Holt-Winters Multiplicative Forecast', color='orange')
plt.title('Holt-Winters Multiplicative Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.show()

# ================
# Feature selection/dimensionality reduction
# ================

numeric_features = climate_data.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('Avg_Temp')

# Standardizing the data
X = climate_data[numeric_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.fillna(0))

# Applying SVD
svd = TruncatedSVD(n_components=min(X_scaled.shape) - 1)
X_svd = svd.fit_transform(X_scaled)

# Analyzing the explained variance
explained_variance = svd.explained_variance_ratio_.cumsum()

# Choosing the number of components that explain, e.g., > 95% of the variance
n_components = np.argmax(explained_variance >= 0.95) + 1
print(f"Number of components explaining >95% variance: {n_components}")

# Reconstructing the data with the selected components
X_reduced = X_svd[:, :n_components]

# Print the reduced dataset
print("Reduced dataset using SVD:")
print(X_reduced)

# ===============
# Base-models
# ===============
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np

h = len(test_data)  # h-step prediction

# Average Model
y_hat_avg = np.mean(train_data)
forecast_avg = np.full(shape=h, fill_value=y_hat_avg)

# Naive Model
y_hat_naive = train_data.iloc[-1]
forecast_naive = np.full(shape=h, fill_value=y_hat_naive)

# Drift Model
y_hat_drift = train_data.iloc[-1] + np.arange(1, h+1) * ((train_data.iloc[-1] - train_data.iloc[0]) / (len(train_data) - 1))
forecast_drift = y_hat_drift

# Simple Exponential Smoothing
ses_model = SimpleExpSmoothing(train_data).fit()
forecast_ses = ses_model.forecast(h)

# Calculate errors for each model
mse_avg = mean_squared_error(test_data, forecast_avg)
mse_naive = mean_squared_error(test_data, forecast_naive)
mse_drift = mean_squared_error(test_data, forecast_drift)
mse_ses = mean_squared_error(test_data, forecast_ses)

print(f"Average Model MSE: {mse_avg}")
print(f"Naive Model MSE: {mse_naive}")
print(f"Drift Model MSE: {mse_drift}")
print(f"SES Model MSE: {mse_ses}")

# ============================
# Multiple Linear Regression
# ============================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
import numpy as np

# Split the data into training and testing sets
y = climate_data['Avg_Temp_Filled']
train_size = int(0.8 * len(X_reduced))
X_train, X_test = X_reduced[:train_size], X_reduced[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

# Model summary using statsmodels for detailed statistics
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()

print("Multiple Linear Regression Model Performance:")
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
print("Adjusted R-squared:", model_sm.rsquared_adj)
print(model_sm.summary())

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(X_reduced):
    X_train_cv, X_test_cv = X_reduced[train_index], X_reduced[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]

    model_cv = LinearRegression()
    model_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = model_cv.predict(X_test_cv)

    mse_cv = mean_squared_error(y_test_cv, y_pred_cv)
    print(f"MSE for fold: {mse_cv}")


from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

residuals = y_test - y_pred_test

# ACF plot of residuals
plot_acf(residuals, lags=40)
plt.show()

# Durbin-Watson statistic (for autocorrelation in residuals)
dw_statistic = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw_statistic}")

# Variance and mean of the residuals
residuals_variance = np.var(residuals)
residuals_mean = np.mean(residuals)
print(f"Variance of residuals: {residuals_variance}")
print(f"Mean of residuals: {residuals_mean}")



plt.figure(figsize=(12, 6))
plt.plot(y_train.index, y_train, label='Train Actual')
plt.plot(y_test.index, y_test, label='Test Actual')
plt.plot(y_test.index, y_pred_test, label='Test Predicted')
plt.title('Linear Regression Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.show()


# ============================
# ARMA / ARIMA / SARIMA / Multiplicative Model
# ============================

# =====
# acf and pacf plots
# =====
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

y = climate_data['Avg_Temp_Filled'].dropna()

# ACF and PACF plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(y, lags=40,ax=ax1, title='Autocorrelation Function')
plot_pacf(y, lags=40,  ax=ax2,title='Partial Autocorrelation Function')
plt.show()


# =====
# GPAC Table for ARMA Order Determination:
# =====

from statsmodels.tsa.stattools import pacf
import pandas as pd

def calculate_gpac(data, nlags, max_ar, max_ma):
    pacf_vals = pacf(data, nlags=nlags)
    gpac_table = pd.DataFrame(index=range(1, max_ar + 1), columns=range(max_ma + 1))

    for k in range(1, max_ar + 1):
        for j in range(max_ma + 1):
            if j == 0:
                gpac_table.loc[k, j] = pacf_vals[k]
            else:
                gpac_table.loc[k, j] = (pacf_vals[k+j] - pacf_vals[j-1]*pacf_vals[k]) / (pacf_vals[k] - pacf_vals[j-1]*pacf_vals[k-j])

    return gpac_table

# Calculate GPAC table
gpac_table = calculate_gpac(y, nlags=40, max_ar=10, max_ma=10)

plt.figure(figsize=(10, 8))
sns.heatmap(gpac_table.astype(float), annot=True, cmap="coolwarm", center=0)
plt.title('GPAC Table')
plt.xlabel('MA Order (q)')
plt.ylabel('AR Order (p)')
plt.show()

# ========
# Arma
# ========

from statsmodels.tsa.arima.model import ARIMA

arma_model = ARIMA(y_train, order=(4, 0, 4))
arma_result = arma_model.fit()
print("ARMA Model Summary:\n\n")
print(arma_result.summary())
predictions = arma_result.get_forecast(steps=len(test_data))
predicted_mean = predictions.predicted_mean
predicted_conf_int = predictions.conf_int()
plt.figure(figsize=(15, 6))
plt.plot(train_data.index, train_data, label='Train Data', color='blue')
plt.plot(test_data.index, test_data, label='Test Data', color='green')
plt.plot(predicted_mean.index, predicted_mean, label='Predicted', color='red')
plt.fill_between(predicted_conf_int.index,
                 predicted_conf_int.iloc[:, 0],
                 predicted_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('ARMA Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.show()
# ========
# Arima
# ========
arima_model = ARIMA(y_train, order=(4, 1, 4))
arima_result = arima_model.fit()
print("ARIMA Model Summary:\n\n")
print(arima_result.summary())
predictions = arima_result.get_forecast(steps=len(test_data))
predicted_mean = predictions.predicted_mean
predicted_conf_int = predictions.conf_int()
plt.figure(figsize=(15, 6))
plt.plot(train_data.index, train_data, label='Train Data', color='blue')
plt.plot(test_data.index, test_data, label='Test Data', color='green')
plt.plot(predicted_mean.index, predicted_mean, label='Predicted', color='red')
plt.fill_between(predicted_conf_int.index,
                 predicted_conf_int.iloc[:, 0],
                 predicted_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('ARIMA Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.show()


# ========
# Sarima
# ========

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Assuming y_train is your training dataset
sarima_model = SARIMAX(y_train, order=(4, 1, 4), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()
print("SARIMA Model Summary:\n\n")
print(sarima_result.summary())


predictions = sarima_result.get_forecast(steps=len(test_data))
predicted_mean_sarima = predictions.predicted_mean
predicted_conf_int = predictions.conf_int()

mse_sarima = mean_squared_error(y_test, predicted_mean_sarima)
print(f"MSE for SARIMA: {mse_sarima}")


plt.figure(figsize=(15, 6))
plt.plot(train_data.index, train_data, label='Train Data', color='blue')
plt.plot(test_data.index, test_data, label='Test Data', color='green')
plt.plot(predicted_mean_sarima.index, predicted_mean_sarima, label='Predicted', color='red')
plt.fill_between(predicted_conf_int.index,
                 predicted_conf_int.iloc[:, 0],
                 predicted_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('SARIMA Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.show()
# ========
# Estimate arma parameters
# ========
from scipy.optimize import least_squares
def arma_objective_function(params, y, ar_order, ma_order):

    ar_params = params[:ar_order]
    ma_params = params[ar_order:ar_order + ma_order]
    residuals = np.empty(len(y))

    for i in range(len(y)):
        ar_term = np.dot(ar_params, y[max(0, i - ar_order):i][::-1]) if i >= ar_order else 0
        ma_term = np.dot(ma_params, residuals[max(0, i - ma_order):i][::-1]) if i >= ma_order else 0
        residuals[i] = y[i] - ar_term - ma_term

    return residuals[max(ar_order, ma_order):]

def estimate_arma_params(y, ar_order, ma_order, initial_params):

    # Estimates ARMA parameters using the Levenberg-Marquardt algorithm.
    result = least_squares(arma_objective_function, initial_params, args=(y, ar_order, ma_order), method='lm')

    estimated_params = result.x
    residuals = result.fun
    jacobian = result.jac
    cov_matrix = np.linalg.inv(jacobian.T.dot(jacobian))  # Covariance matrix
    error_variance = np.var(residuals, ddof=ar_order + ma_order)  # Variance of the error
    return estimated_params, cov_matrix, error_variance, residuals, result.nfev

y = y_train
ar_order = 4
ma_order = 4
initial_params = np.concatenate((np.random.rand(ar_order) - 0.5, np.random.rand(ma_order) - 0.5))

estimated_params, cov_matrix, error_variance, residuals, iterations = estimate_arma_params(y, ar_order, ma_order,
                                                                                           initial_params)
print("Parameter estimates:", estimated_params)
print("Standard errors:", np.sqrt(np.diag(cov_matrix)))

# Calculate and display the confidence intervals for the parameter estimates

confidence_interval = 1.96 * np.sqrt(np.diag(cov_matrix)) / np.sqrt(len(y))

print("95% confidence intervals:\n", confidence_interval)

# ============
# Residual Analysis
# ============

# a. Whiteness Chi-square test
# We will use Ljung-Box test as an alternative to check the whiteness of residuals
import scipy.stats as stats

ljung_box_sarima_result = acorr_ljungbox(sarima_result.resid, lags=[10], return_df=True)
print(ljung_box_sarima_result)

# b. Display the estimated variance of the error and the estimated covariance of the estimated parameters.
error_variance = np.var(sarima_result.resid, ddof=len(sarima_result.params))
print(f"Estimated variance of the error: {error_variance}")

cov_params = sarima_result.cov_params()
print("Estimated covariance of the estimated parameters:")
print(cov_params)

# c. Is the derived model biased or this is an unbiased estimator?
# Check if the mean of residuals is statistically different from zero
t_stat, p_value = stats.ttest_1samp(sarima_result.resid, 0)
print(f"Mean of residuals: {np.mean(sarima_result.resid)}")

print(f"t-statistic: {t_stat}, p-value: {p_value}")
if p_value < 0.05:
    print("We reject the null hypothesis that the residuals have a mean of zero. The model might be biased.")
else:
    print("We do not reject the null hypothesis that the residuals have a mean of zero. The model is likely unbiased.")

# d. Check the variance of the residual errors versus the variance of the forecast errors.
forecast_errors = y_test - predicted_mean_sarima
forecast_error_variance = np.var(forecast_errors, ddof=1)
print(f"Variance of the forecast errors: {forecast_error_variance}")
if forecast_error_variance > error_variance:
    print("The variance of the forecast errors is greater than the variance of the residual errors.")
else:
    print("The variance of the residual errors is greater than or equal to the variance of the forecast errors.")

conf_int = sarima_result.conf_int()
print("Confidence intervals for the coefficients:")
print(conf_int)
sarima_result.plot_diagnostics(figsize=(15, 12))
plt.show()

# e. Perform zero-pole cancellation operation and display the final coefficient confidence interval.
ar_poly = np.r_[1, -estimated_params[:ar_order]]
ma_poly = np.r_[1, estimated_params[ar_order:]]

poles = np.roots(ar_poly)
zeros = np.roots(ma_poly)

print(f"Poles: {np.array2string(poles, precision=3, separator=', ')}")
print(f"Zeros: {np.array2string(zeros, precision=3, separator=', ')}")

# Check for pole-zero cancellation
for pole in poles:
    if any(np.isclose(pole, zero) for zero in zeros):
        print(f"Pole-zero cancellation detected at: {pole:.3f}")

