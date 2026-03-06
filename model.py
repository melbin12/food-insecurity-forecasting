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

