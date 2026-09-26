"""Example 2: Estimate a simple relationship from data."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Each row is an observation.
X = np.array([[100], [200], [300], [400], [500]])
y = np.array([1200, 1800, 2500, 3200, 3900])

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("MAE:", mean_absolute_error(y, predictions))
print("MSE:", mean_squared_error(y, predictions))

# Mathematical form:
#
# y_hat = a + b*x
