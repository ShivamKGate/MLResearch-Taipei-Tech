import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.inspection import permutation_importance

# Load data
data = pd.read_csv('rmse_test.csv')

# Prepare data
X = data[['epoch']]
y = data['rmse']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X, y)

# Predict RMSE for future overall epoch tests
future_epochs = [[15], [16],] 
predicted_rmse = model.predict(future_epochs)
print(f'Predicted RMSE for epochs 15 and 16: {predicted_rmse}')

# Scatter plot with regression line for Linear Regression
plt.figure(figsize=(8, 6))
sns.regplot(x='epoch', y='rmse', data=data, scatter_kws={'s': 50, 'alpha': 0.5})
plt.title('Deviation of RMSE Over Epochs (Linear Regression)')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.show()

# Hyperparameter tuning for Ridge regression
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Grid search with cross-validation for Ridge regression
ridge_model = Ridge()
grid_search = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print("Ridge Regression - Best Parameter:", grid_search.best_params_)
print("Ridge Regression - Best Negative MSE:", grid_search.best_score_)

# Evaluate best Ridge model on test set
best_ridge_model = grid_search.best_estimator_
y_pred_ridge = best_ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print("Mean Squared Error (Ridge Regression):", mse_ridge)

# Learning curve for linear regression
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)

plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Example of cross-validation RMSE scores
scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print("Cross-Validation RMSE scores:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())

# Train a Random Forest model
forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)

# Calculate Mean Squared Error (Random Forest)
mse_forest = mean_squared_error(y_test, y_pred_forest)
print("Mean Squared Error (Random Forest):", mse_forest)

# Assuming predictions and actuals are available
predictions = model.predict(X)
residuals = y - predictions

plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted RMSE')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.show()

# autocorrelation plot
plot_acf(data['rmse'],)
plt.show()

#gradient boosting regressor
gb_model = GradientBoostingRegressor()
gb_model.fit(X, y)



