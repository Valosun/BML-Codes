import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

plt.scatter(X, y, color='blue')
plt.title("Synthetic Linear Regression Data")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.grid(True)
plt.show()

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title("Linear Regression Fit")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.grid(True)
plt.show()

print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)


