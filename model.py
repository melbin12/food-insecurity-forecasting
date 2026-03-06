# Install Prophet (only if not already installed)
!pip install prophet --quiet

# Import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Time Series Models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Others
import warnings
warnings.filterwarnings("ignore")

# Replace the path below with your actual file path
data_path = '/content/IPC_IPC.csv'

# Load the dataset
df = pd.read_csv(data_path)

# Preview the dataset
df.head()
# Overview of the data
df.info()

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Step 1: Select relevant columns
df_clean = df[['REF_AREA_LABEL', 'TIME_PERIOD', 'OBS_VALUE']].copy()

# Step 2: Rename columns for simplicity
df_clean.columns = ['country', 'date', 'ipc_p3plus']

# Step 3: Convert date column to datetime
df_clean['date'] = pd.to_datetime(df_clean['date'], format='%Y-%m')

# Step 4: Drop rows with missing ipc values
df_clean = df_clean.dropna(subset=['ipc_p3plus'])

# Step 5: Sort data
df_clean = df_clean.sort_values(by=['country', 'date']).reset_index(drop=True)

# Step 6: Preview cleaned data
df_clean.head()

# Group by country and date, summing ipc_p3plus values
df_agg = df_clean.groupby(['country', 'date'], as_index=False)['ipc_p3plus'].sum()

# Display the first 10 rows
df_agg.head(10)
# Filter data for one country, e.g., Afghanistan
country_name = 'Afghanistan'
df_country = df_agg[df_agg['country'] == country_name]

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(df_country['date'], df_country['ipc_p3plus'], marker='o')
plt.title(f"IPC Phase 3+ Population Over Time: {country_name}")
plt.xlabel("Date")
plt.ylabel("People in IPC Phase 3+")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Check types and value examples
print(df_agg.dtypes)

# Check for any non-numeric examples
non_numeric = df_agg[~df_agg['ipc_p3plus'].apply(lambda x: isinstance(x, (int, float)))]
print("Non-numeric rows:", non_numeric.shape[0])
print(non_numeric.head())
# Count observations per country
obs_counts = df_agg.groupby('country').size().sort_values()

# Show countries with fewest data points
print(obs_counts.head(20))
# Count observations per country
obs_counts = df_agg.groupby('country').size()

# Keep only countries with at least 12 records
eligible_countries = obs_counts[obs_counts >= 12].index.tolist()

print(f"{len(eligible_countries)} countries have enough data.")
print("Examples:", eligible_countries[:5])
from statsmodels.tsa.arima.model import ARIMA

all_forecasts = []

for country in eligible_countries:
    try:
        # Filter country data
        df_country = df_agg[df_agg['country'] == country].copy()

        # Set date as index
        df_country = df_country.set_index('date')

        # IMPORTANT FIX: resample ONLY the numeric series
        ts = df_country['ipc_p3plus']
        ts = ts.resample('MS').mean()
        ts = ts.ffill()

        # Skip if still too short
        if ts.count() < 12:
            print(f"Skipping {country}: insufficient data after resampling")
            continue

        # Fit ARIMA
        model = ARIMA(ts, order=(1, 1, 1))
        model_fit = model.fit()

        # Forecast next 24 months
        forecast_steps = 24
        forecast = model_fit.forecast(steps=forecast_steps)

        # Create future dates
        future_dates = pd.date_range(
            start=ts.index[-1] + pd.DateOffset(months=1),
            periods=forecast_steps,
            freq='MS'
        )

        # Store forecast
        df_forecast = pd.DataFrame({
            'country': country,
            'date': future_dates,
            'forecast': forecast.values
        })

        all_forecasts.append(df_forecast)
        print(f"✅ Success: {country}")

    except Exception as e:
        print(f"❌ Error for {country}: {e}")

# Combine forecasts
if all_forecasts:
    df_all_forecasts = pd.concat(all_forecasts, ignore_index=True)
    print("\nForecast sample:")
    print(df_all_forecasts.head())
else:
    print("No successful forecasts.")

