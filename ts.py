import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


# Load the data
data = pd.read_csv('rmse_test.csv')
data['epoch'] = pd.to_datetime(data['epoch'])

# Set 'epoch' as the index for time series analysis
data.set_index('epoch', inplace=True)

# Replace inf values with NaN explicitly
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Extracting features and target
X = data.index.values.reshape(-1, 1)  
y = data['rmse'].values

# TimeSeries split for cross-validation
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Plot RMSE over time
plt.figure(figsize=(10, 6))
sns.lineplot(x=data.index, y='rmse', data=data)
plt.title('RMSE Over Time')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.grid(True)
plt.show()

