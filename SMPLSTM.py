import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
data = pd.read_csv('/Users/mac/Desktop/Projects/SMPLSTM/TSLA.csv')  # Replace 'stock_data.csv' with your dataset file

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences for LSTM training
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # Number of previous time steps to consider for each prediction
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("Training the LSTM model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
print("Evaluating the model...")
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print("Train Loss:", train_loss)
print("Test Loss:", test_loss)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Plot the predicted vs. actual prices
plt.plot(scaler.inverse_transform(y_train), label='Actual Train Prices')
plt.plot(scaler.inverse_transform(train_predictions), label='Predicted Train Prices')
plt.plot(scaler.inverse_transform(y_test), label='Actual Test Prices')
plt.plot(scaler.inverse_transform(test_predictions), label='Predicted Test Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Market Prediction with LSTM')
plt.legend()
plt.show()
