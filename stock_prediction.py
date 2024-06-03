import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Deep Learning Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the data
file_path = 'GOOG.csv'
df = pd.read_csv(file_path)
print(df.head())
print(df.info())

# Data Preparation
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Feature Engineering: Use previous day's closing price to predict the next day's closing price
df['Prev Close'] = df['Close'].shift(1)
df.dropna(inplace=True)

# Create features and labels
X = df[['Prev Close']]
y = df['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate the model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_test, y_pred

# Linear Regression Model
lr_model = LinearRegression()
lr_mse, lr_r2, y_test_lr, y_pred_lr = evaluate_model(lr_model, X_train, y_train, X_test, y_test)

# Decision Tree Model
dt_model = DecisionTreeRegressor()
dt_mse, dt_r2, y_test_dt, y_pred_dt = evaluate_model(dt_model, X_train, y_train, X_test, y_test)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100)
rf_mse, rf_r2, y_test_rf, y_pred_rf = evaluate_model(rf_model, X_train, y_train, X_test, y_test)

# Support Vector Machine Model
svr_model = SVR()
svr_mse, svr_r2, y_test_svr, y_pred_svr = evaluate_model(svr_model, X_train, y_train, X_test, y_test)

# Long Short-Term Memory (LSTM) Model
# Reshape data for LSTM
X_train_lstm = np.array(X_train).reshape(-1, 1, 1)
X_test_lstm = np.array(X_test).reshape(-1, 1, 1)

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)
y_pred_lstm = lstm_model.predict(X_test_lstm)
lstm_mse = mean_squared_error(y_test, y_pred_lstm)
lstm_r2 = r2_score(y_test, y_pred_lstm)

# Print evaluation metrics
print(f"Linear Regression MSE: {lr_mse}, R2: {lr_r2}")
print(f"Decision Tree MSE: {dt_mse}, R2: {dt_r2}")
print(f"Random Forest MSE: {rf_mse}, R2: {rf_r2}")
print(f"Support Vector Machine MSE: {svr_mse}, R2: {svr_r2}")
print(f"LSTM MSE: {lstm_mse}, R2: {lstm_r2}")

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred_lr, label='Linear Regression', linestyle='--', color='orange')
plt.plot(y_pred_dt, label='Decision Tree', linestyle='--', color='green')
plt.plot(y_pred_rf, label='Random Forest', linestyle='--', color='red')
plt.plot(y_pred_svr, label='Support Vector Machine', linestyle='--', color='purple')
plt.plot(y_pred_lstm, label='LSTM', linestyle='--', color='brown')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
