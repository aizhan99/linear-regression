import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

with open('data6.txt', 'r') as file:
    data = file.readlines()

features = [[] for _ in range(13)]
target = []

for line in data:
    values = line.strip().split()
    for i in range(13):
        features[i].append(float(values[i]))
    target.append(float(values[13]))

X = np.array(features).T  # Transpose to have (number of samples, number of features)

X_train, X_test, Y_train, Y_test = train_test_split(X, target, test_size=0.2, random_state=42)

covariance_values = [np.cov(X_train[:, i], Y_train)[0, 1] for i in range(13)]

correlation_values = [np.corrcoef(X_train[:, i], Y_train)[0, 1] for i in range(13)]

for i in range(13):
    print(f"Covariance between X{i+1} and Y: {covariance_values[i]}")
    print(f"Correlation between X{i+1} and Y: {correlation_values[i]}")

import pandas as pd

data = {
    "Feature": [f"X{i + 1}" for i in range(13)],
    "Covariance": covariance_values,
    "Correlation": correlation_values
}

df = pd.DataFrame(data)

print(df)

std_deviation_features = [np.std(X_train[:, i], ddof=1) for i in range(13)]

std_deviation_target = np.std(Y_train, ddof=1)

values = [covariance_values[i] / (std_deviation_features[i] * std_deviation_target) for i in range(13)]

for i in range(13):
    print(f"Absolute value of correlation for X{i+1}: {values[i]}")

import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = np.corrcoef(X_train.T)

plt.figure(figsize=(10, 8))
tick_labels = [f"X{i + 1}" for i in range(13)]
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=tick_labels, yticklabels=tick_labels)
plt.title("Correlation Matrix")
plt.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, Y_train)

beta0 = model.intercept_
beta_i = model.coef_

data = {
    "Coefficient": ["Intercept (β0)"] + [f"β{i + 1}" for i in range(len(beta_i))],
    "Value": [beta0] + list(beta_i)
}

df = pd.DataFrame(data)
print(df)

from sklearn.metrics import mean_squared_error

predicted_Y = model.predict(X_train)

mse = mean_squared_error(Y_train, predicted_Y)

estimated_variance_of_noise = mse

print("Estimated Variance of Noise (σ²) using MSE:", estimated_variance_of_noise)

from numpy.linalg import inv

coefficients = model.coef_

residuals = Y_train - model.predict(X_train)

n = len(Y_train)
p = len(coefficients)

rss = np.sum(residuals ** 2)
mse = rss / (n - p - 1)
se = np.sqrt(mse)

X_transpose_X = np.dot(X_train.T, X_train)
X_transpose_X_inv = inv(X_transpose_X)
t_values = coefficients / (se * np.sqrt(np.diag(X_transpose_X_inv)))

data = {
    "Predictor": [f"X{i + 1}" for i in range(p)],
    "T-Value": t_values
}
df = pd.DataFrame(data)
print(df)

alpha = 2

selected_features = []

for i in range(len(t_values)):
    if abs(t_values[i]) > alpha:
        selected_features.append(f"X{i + 1}")

print("Relevant features based on t-values and alpha:", selected_features)

import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

model = sm.OLS(Y_train, sm.add_constant(X_train)).fit()

predicted_Y = model.predict()

residuals = Y_train - predicted_Y

fig, ax = plt.subplots(figsize=(6, 4))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()

plt.scatter(predicted_Y, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values (Y-hat)")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Values")
plt.show()

r_squared = model.rsquared

print("R-squared:", r_squared)