import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the Boston dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='PRICE')

# Display the dataset
print("First 5 rows of the dataset:")
print(X.head())
print("\nTarget variable (House Prices):")
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a Simple Linear Regression Model (Using RM feature)
X_train_rm = X_train[['RM']]  # RM: Average number of rooms per dwelling
X_test_rm = X_test[['RM']]

simple_model = LinearRegression()
simple_model.fit(X_train_rm, y_train)

# Predict and evaluate the Simple Linear Regression Model
y_pred_simple = simple_model.predict(X_test_rm)
simple_rmse = np.sqrt(mean_squared_error(y_test, y_pred_simple))
simple_r2 = r2_score(y_test, y_pred_simple)

print("\nSimple Linear Regression Model:")
print(f"Root Mean Squared Error (RMSE): {simple_rmse}")
print(f"R-squared (R2): {simple_r2}")

# Plot the regression line for RM
plt.scatter(X_test_rm, y_test, color='blue', label='Actual Prices')
plt.plot(X_test_rm, y_pred_simple, color='red', label='Regression Line')
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Price")
plt.title("Simple Linear Regression: RM vs Price")
plt.legend()
plt.show()

# Build a Multiple Regression Model (Using all features)
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

# Predict and evaluate the Multiple Regression Model
y_pred_multi = multi_model.predict(X_test)
multi_rmse = np.sqrt(mean_squared_error(y_test, y_pred_multi))
multi_r2 = r2_score(y_test, y_pred_multi)

print("\nMultiple Regression Model:")
print(f"Root Mean Squared Error (RMSE): {multi_rmse}")
print(f"R-squared (R2): {multi_r2}")

# Display model coefficients
coefficients = pd.DataFrame(multi_model.coef_, index=X.columns, columns=['Coefficient'])
print("\nCoefficients of the Multiple Regression Model:")
print(coefficients)
