

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Walmart.xls")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)



print("\nMissing Values:")
print(df.isnull().sum())


df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')


df = df.dropna(subset=['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day


print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())



plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Weekly_Sales'])
plt.title("Weekly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.grid(True)
plt.show()



store_sales = df.groupby('Store')['Weekly_Sales'].sum()

plt.figure(figsize=(12,6))
store_sales.plot(kind='bar')
plt.title("Total Sales by Store")
plt.xlabel("Store")
plt.ylabel("Total Sales")
plt.show()



X = df[['Store',
        'Holiday_Flag',
        'Temperature',
        'Fuel_Price',
        'CPI',
        'Unemployment',
        'Year',
        'Month',
        'Day']]

y = df['Weekly_Sales']



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Training Completed!")



y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MAE     :", mae)
print("MSE     :", mse)
print("RMSE    :", rmse)
print("R2 Score:", r2)



plt.figure(figsize=(12,6))
plt.plot(y_test.values[:100], label="Actual Sales")
plt.plot(y_pred[:100], label="Predicted Sales")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Records")
plt.ylabel("Weekly Sales")
plt.legend()
plt.show()


future_data = pd.DataFrame({
    'Store': [1],
    'Holiday_Flag': [0],
    'Temperature': [70],
    'Fuel_Price': [3.5],
    'CPI': [220],
    'Unemployment': [6.0],
    'Year': [2026],
    'Month': [12],
    'Day': [1]
})

future_sales = model.predict(future_data)

print("\nPredicted Future Sales:")
print(future_sales[0])


predictions = pd.DataFrame({
    'Actual_Sales': y_test,
    'Predicted_Sales': y_pred
})


predictions.to_excel("sales_predictions.xlsx", index=False)

print("\nPrediction file saved successfully!")

