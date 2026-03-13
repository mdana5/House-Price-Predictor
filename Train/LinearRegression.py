import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Processed Dataset
df = pd.read_csv("housingfinalprocessed.csv")

# 2. Separate Features and Target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Convert boolean values to integers if present
X = X.astype(int)


# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize Linear Regression
lr_model = LinearRegression()


# 5. Train Model

lr_model.fit(X_train, y_train)

# 6. Predictions
y_pred = lr_model.predict(X_test)


# 7. Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Model Performance")
print("-----------------------------------")
print("MAE  :", mae)
print("RMSE :", rmse)
print("R2   :", r2)

# 8. Save Model
joblib.dump(lr_model, "linear_regression_model.pkl")


# 9. Save Column Structure
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("\nModel saved successfully!")
print("Saved files:")
print(" - linear_regression_model.pkl")
print(" - model_columns.pkl")