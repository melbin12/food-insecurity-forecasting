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
